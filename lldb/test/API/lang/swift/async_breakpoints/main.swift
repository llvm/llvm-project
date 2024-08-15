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
}

await foo()
