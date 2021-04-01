@MainActor func stringNum(_ i: Int) async -> String {
  let x = await randInt(i)
  let y = await randInt(i)
  return String(x + y)
}

@MainActor func randInt(_ i: Int) async -> Int {
  return Int.random(in: 1...i)
}

func use<T>(_ t: T...) {}

@main struct Main {
  static func main() async {
    let a = await stringNum(30)
    let b = await stringNum(41)
    use(a, b)
  }
}
