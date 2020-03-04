import CFoo
import Foundation

func use<T>(_ t: T) {}

@objc public class Foo : NSObject {
  @objc public func f() {
    let foo = CFoo(x: 23)
    let bar = FromBridgingHeader(y: 42)
    use(foo) // break here
    use(bar)
  }
}
