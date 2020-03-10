import CFoo
import Foundation

func use<T>(_ t: T) {}

@objc public class Foo : NSObject {
  @objc public func f() {
    let baz = Baz(i_am_from_Foo: 23)
    use(baz) // break here
  }
}
