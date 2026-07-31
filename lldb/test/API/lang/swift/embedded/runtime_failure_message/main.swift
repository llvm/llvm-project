// The message is folded into the trap only where the builtin sees a literal, so
// each case depends on whether doomed is inlined into its caller.
#if FALLBACK
// Not inlined: the caller's literal never reaches the builtin, and a generic
// message is substituted for it.
@inline(never)
#else
// Inlined: the builtin sees the caller's literal and folds it into the trap.
@inline(__always)
#endif
func doomed(_ x: Int, _ message: StaticString) -> Int {
#if ASSERT
  assert(x >= 0, message)
#elseif ASSERTION_FAILURE
  if x < 0 {
    assertionFailure(message)
  }
#elseif PRECONDITION
  precondition(x >= 0, message)
#elseif PRECONDITION_FAILURE
  if x < 0 {
    preconditionFailure(message)
  }
#else
  if x < 0 {
    fatalError(message)
  }
#endif
  return x
}

_ = doomed(-1, "index must not be negative")
