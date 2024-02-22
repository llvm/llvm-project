@main struct Main {
  static func main() async {
    let main = Main()
    await main.main0()
  }

  func main0() async {
    await main1("ignored")
  }
}

func main1<T>(_ unused: T) async {
  let a = await main2(30)
  let b = await main3(a + 11)
  use(a, b)
}

func main2(_ i: Int) async -> Int {
  return Int.random(in: 1...Int(i))
}

@MainActor func main3(_ i: Int) async -> Int {
  return Int.random(in: 1...i)
}

func use<T>(_ t: T...) {}

