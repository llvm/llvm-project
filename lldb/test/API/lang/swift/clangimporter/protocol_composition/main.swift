import Foundation
@objc public protocol MyProtocol : NSObjectProtocol {}

@objc class C : NSObject {
  let i = 23
}

extension C : MyProtocol {
}

class D {
  let p : MyProtocol = C()
}

func use<T>(_ t: T) {}

func main() {
  let obj = D()
  use(obj) // break here
}

main()
