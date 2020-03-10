import Foundation

func use<T>(_ t: T) {}

@objc public class Foo: NSObject {
  let i = 23
  override public init() {}
  @objc public func f() {
    use(i) // break here
  }
}
