public class F<A> {
  let a : A
  var b = 42
  var c = 128
  var d = 256
  public init(_ val : A) { a = val }
}

protocol P {
  func method()
}
extension F : P {
  @inline(never) func method() {
    print("break here \(b) \(self)")
  }
}

// Defeat type specialization.
@inline(never) func getF() -> P {
  return F<Int>(23)
}

let obj = getF()
obj.method()

