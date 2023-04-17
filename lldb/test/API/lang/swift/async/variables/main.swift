import Swift
func use<T>(_ t: T) {}
func sink<T>(_ t: T) {}

func split() async {}

func f(_ xs: [Int?]) async {
  for x in xs {
    let x = x!
    sink(x) // break here
  }
  await split()
  for x in xs {
    let x = x!
    sink(x) // break here
  }
}

@main struct Main {
  static func main() async {
    await f([23])
  }
}
