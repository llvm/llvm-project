func work(_ objects: Any...) {
  for object in objects {
    print("Processing object of type: \(type(of: object))")
  }
}

func use(_ x: Int, _ y: Int) -> Int {
    return x &* y &+ x &- y
}

var arr: [Int] = []

func ASYNC___1___() async -> Int {
  var a1 = 1, a2 = 2, a3 = 3, a4 = 4, a5 = 5
  print("BREAK HERE")
  var a6 = 6, a7 = 7, a8 = 8, a9 = 9, a10 = 10
  var a11 = 11, a12 = 12, a13 = 13, a14 = 14, a15 = 15
  var a16 = 16, a17 = 17, a18 = 18, a19 = 19, a20 = 20
  var a21 = 21, a22 = 22, a23 = 23, a24 = 24, a25 = 25
  var a26 = 26, a27 = 27, a28 = 28, a29 = 29, a30 = 30
  a1 = use(a1, a2) 
  a3 = use(a3, a4)
  a5 = use(a5, a6)
  a7 = use(a7, a8)
  a9 = use(a9, a10)
  a11 = use(a11, a12)
  a13 = use(a13, a14)
  a15 = use(a15, a16)
  a17 = use(a17, a18)
  a19 = use(a19, a20)
  a21 = use(a21, a22)
  a23 = use(a23, a24)
  a25 = use(a25, a26)
  a27 = use(a27, a28)
  a29 = use(a29, a30)
  work(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24,a25, a26,a27, a28, a29, a30)
  arr = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24,a25, a26,a27, a28, a29, a30]
  return a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 + a16 + a17 + a18 + a19
}

func ASYNC___2___() async -> Int {
  let result = await ASYNC___1___()
  return result
}

func ASYNC___3___() async -> Int {
  let result = await ASYNC___2___()
  return result
}

func ASYNC___4___() async -> Int {
  let result = await ASYNC___3___()
  return result
}

func ASYNC___5___() async -> Int {
  let result = await ASYNC___4___()
  return result
}

@main struct Main {
  static func main() async {
    let result = await ASYNC___5___()
    print(result)
  }
}
