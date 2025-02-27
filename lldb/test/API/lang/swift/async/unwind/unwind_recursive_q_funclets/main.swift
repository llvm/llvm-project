func factorial(_ n: Int) async -> Int {
  if (n == 0) {
    return 1;
  }
  let n1 = await factorial(n - 1)
  return n * n1
}

@main struct Main {
  static func main() async {
    let result = await factorial(10)
    print(result)
  }
}
