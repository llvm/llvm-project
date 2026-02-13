func USE(_ int: inout Int) {} // USE line
func FINISH() {}

struct S {
  private var member_int: Int = 42

  var COMPUTED_PROPERTY: Int {
    get { return member_int }
    yielding mutate {
      FINISH() // first coroutine line
      yield &member_int //yield line
      FINISH() // last coroutine line
    }
  }
}

var state = S()
USE(&state.COMPUTED_PROPERTY) // coroutine call site
FINISH() // last main line
