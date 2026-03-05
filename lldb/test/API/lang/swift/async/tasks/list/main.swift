func factorial(n: Int) async -> Int {
  if n == 1 {
    return 1  //break here
  }
  async let n1 = factorial(n: n - 1)

  return await n1 * n
}

@main struct Main {
  static func main() async {
    let task = Task (name: "factorial-main") {
      await factorial(n: 3)
    }
    await task.value
  }
}
