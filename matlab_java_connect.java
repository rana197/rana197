package matlab;

import com.mathworks.engine.*;
import java.util.concurrent.*;
import java.util.logging.Level;
import java.util.logging.Logger;
public class matlab_Topsis_Test {
	private static MatlabEngine eng;
	public static void main(String[] args) { 
		try {
		eng = MatlabEngine.startMatlab();
		eng.eval("x= [-20:0.2:20];");
		eng.eval("y= x.^2;");
		eng.eval("plot(x,y)");
		Thread.sleep(10000);
		} catch (InterruptedException ex) {
			Logger.getLogger(matlab_Topsis_Test.class.getName()).log(Level.SEVERE,null,ex);
		} catch (EngineException ex) {
			Logger.getLogger(matlab_Topsis_Test.class.getName()).log(Level.SEVERE,null,ex);
		} catch (MatlabExecutionException ex) {
			Logger.getLogger(matlab_Topsis_Test.class.getName()).log(Level.SEVERE,null,ex);
		} catch (MatlabSyntaxException ex) {
			Logger.getLogger(matlab_Topsis_Test.class.getName()).log(Level.SEVERE,null,ex);
		} catch (ExecutionException ex) {
			Logger.getLogger(matlab_Topsis_Test.class.getName()).log(Level.SEVERE,null,ex);
		}	 
	}
}

