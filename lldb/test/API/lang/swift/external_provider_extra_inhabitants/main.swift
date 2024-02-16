import CoreGraphics

func use<T>(_ t: T) {}

public class MyClass {
  // CGImage is consulted through the external info provider.
  // This field is here to test that LLDB correctly calculates 
  // that the size of "CGImage?" is 8 bytes, given that CGImage
  // has non-zero extra inhabitants.
	public var unused: CGImage? = nil
	public var size: CGSize? = nil
}
func f() {
  let object = MyClass()
  object.size = CGSize(width:10, height:20)
  use(object)
  print("Set breakpoint here.")
}

f()
