import CoreGraphics

public class MyClass {
  // CGImage is consulted through the external info provider.
  // This field is here to test that LLDB correctly calculates 
  // that the size of "CGImage?" is 8 bytes, given that CGImage
  // has non-zero extra inhabitants.
	public var unused: CGImage? = nil
	public var size: CGSize? = nil
}
let object = MyClass()
object.size = CGSize(width:10, height:20)
print(1) // Set breakpoint here.

