@main enum entry {
  static func main() async {
    print("Breakpoint main")
    async let fib5_task = fib(n: 5)
    async let fib6_task = fib(n: 6)
    let fib4 = await fib(n: 4)
    let fib5 = await fib5_task
    let fib6 = await fib6_task
    print(fib4, fib5, fib6)
  }
}

func fib(n: Int) async -> Int {
  if (n == 0) {
    return 1
  }
  if (n == 1) {
    return 1
  }
  async let n1_task = fib(n: n - 1)  // Breakpoint fib
  async let n2_task = fib(n: n - 2)
  let n1 = await n1_task
  let n2 = await n2_task
  return n1 + n2
}
