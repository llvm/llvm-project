@inline(never)
func blackhole(_ x: Int) {
  if x == -12345 {
    print(x)
  }
}

// A free function with no parameters and no local variables (and no `self`).
@inline(__always)
func freeFunction() {
  blackhole(42) // break here in free function
}

class Tester {
  var count = 41

  // An inlined *method* with its own `self`.
  // This is `private` matters, otherwise the compiler also emits an out-of-line copy
  // of the method.
  @inline(__always)
  private func inlinedMethod() {
    blackhole(count) // break here in method
  }

  @inline(never)
  func testTop() {
    freeFunction()
    inlinedMethod()
    blackhole(count)
  }
}

Tester().testTop()
