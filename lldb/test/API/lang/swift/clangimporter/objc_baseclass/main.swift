import Foundation

class A {
  public static var shared = A()
  let a = 42
}

class C : NSObject {
  var c = 0
  func foo() {
    print("break here")
    c = 1
  }
}

C().foo()

