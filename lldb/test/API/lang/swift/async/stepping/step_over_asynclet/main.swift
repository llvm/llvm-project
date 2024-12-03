func getTimestamp(x: Int) async -> Int {
  return 40 + x
}

func work() {}

func foo() async {
  do {
    work() // BREAK HERE
    async let timestamp1 = getTimestamp(x:1)
    work()
    async let timestamp2 = getTimestamp(x:2)
    work()
    let timestamps = await [timestamp1, timestamp2]
    print(timestamps)
  }
  async let timestamp3 = getTimestamp(x:3)
  work()
  let actual_timestamp3 = await timestamp3
  print(actual_timestamp3)
}

@main enum entry {
  static func main() async {
    await foo()
  }
}
