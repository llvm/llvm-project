func randInt(_ i: Int) async -> Int {
  return Int.random(in: 1...i)
}

func inner<T>(_ t: T) async {
  // d is dynamically allocated by swift_task_alloc() because its size
  // is unknown.
  let d = t
  let a = await randInt(30)
  print("break one")
  let b = await randInt(a + 11)
  use(a, b)
  use(d)
  print("break two")
}

func use<T>(_ t: T...) {}

@main struct Main {
  static func main() async {
    // This call to `inner` is a indirection required to make this test work.
    // If its contents were inlined into `main` (as it was originally written),
    // the test would fail.
    await inner(23)
  }
}
