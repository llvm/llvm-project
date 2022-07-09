func randInt(_ i: Int) async -> Int {
  return Int.random(in: 1...i)
}

func inner() async {
  let a = await randInt(30)
  let b = await randInt(a + 11) // break one
  use(a, b) // break two
}

func use<T>(_ t: T...) {}

@main struct Main {
  static func main() async {
    // This call to `inner` is a indirection required to make this test work.
    // If its contents were inlined into `main` (as it was originally written),
    // the test would fail.
    await inner()
  }
}
