@main enum entry {
  static func main() async {
    let x = await f()
    print(x)
  }
}

func f() async -> Int {
  return 30
}
