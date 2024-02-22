public class Foo {
  public static func foo() {
    print("first")
  }
}

public struct MyPoint {
  let x: Int
  public let y: Int


  public init(x: Int, y: Int) {
    self.x = x
    self.y = y
  }

  public var magnitudeSquared: Int {
    return x*x + y*y
  }
}
