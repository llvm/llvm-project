func getTimestamp(i:Int) async -> Int {
  return i
}

func work() {}

func foo() async {
  work()
  let timestamp1 = await getTimestamp(i:42) // Breakpoint1
  work() // Breakpoint2
  let timestamp2 = await getTimestamp(i:43) // Breakpoint3
  work()
  // There should be two breakpoints below in an async let:
  // One for the code in the "callee", i.e., foo.
  // One for the implicit closure in the RHS.
  async let timestamp3 = getTimestamp(i: 44) // Breakpoint4
  // There should be one breakpoint in an await of an async let variable
  await timestamp3 // Breakpoint5
}

await foo()
