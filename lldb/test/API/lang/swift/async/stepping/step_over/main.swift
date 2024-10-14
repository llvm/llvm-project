@main enum entry {
  static func main() async {
    let x = await f() // BREAK HERE
    print(x)
    for i in 1...2 {
      await f()
    }
    if (await f() == 30) {
      print("here!")
    }
  }
}

func f() async -> Int {
  return 30
}
