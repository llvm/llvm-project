actor Fibonacci {
    var _cache = [ 0, 1 ]

    func fibonacci(_ n : Int) async -> Int {
         if n < _cache.count { // function start
            return _cache[n] // end recursion
         }
         let n_1 = await fibonacci(n - 1)
         let n_2 = await fibonacci(n - 2) // recurse
         print(n, n_1, n_2)
         let res = n_1 + n_2 // compute result
         assert(n == _cache.count)
         _cache.append(res)
         return res
    }
}


func fibonacci(_ n: Int) async -> Int {
    if n == 0  || n == 1 { // function start
        return n // end recursion
    }
    let n_1 = await fibonacci(n - 1)
    let n_2 = await fibonacci(n - 2) // recurse
    print(n, n_1, n_2)
    return n_1 + n_2 // compute result
}

@main struct Main {
  static func main() async {
    let n = await fibonacci(10) // main breakpoint
    print(n)
    let F = Fibonacci()
    let n2 = await F.fibonacci(10) // main actor breakpoint
  }
}
