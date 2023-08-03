actor Actor {
  func f(_ n : Int) async {
    print(n) // break here
  }
}


@main struct Main {
  static func main() async {
    let A = Actor()
    await A.f(42)
  }
}
