func fibonacci(_ n: Int) async -> Int {
    if n == 0  || n == 1 { // function start
        return n // end iteration
    }
    let n_1 = await fibonacci(n - 1)
    let n_2 = await fibonacci(n - 2)
    print(n, n_1, n_2)
    return n_1 + n_2
}

@main struct Main {
  static func main() async {
    let n = await fibonacci(10) // main breakpoint
    print(n)  
  }
}