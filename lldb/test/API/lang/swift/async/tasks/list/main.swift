// Put every task on one serial executor, so each parent must suspend before
// its child can run. That makes the task tree deterministic.
@MainActor
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
