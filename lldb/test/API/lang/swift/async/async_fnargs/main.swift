import Swift
func use<T>(_ t: T) {}

func sayHello() async {
  print("hello")
}

func sayBasic(_ msg: String) async {
  print("Set breakpoint here")
  await sayHello()
  print(msg) // Set breakpoint here
}

func sayGeneric<T>(_ msg: T) async {
  print("Set breakpoint here")
  await sayHello()
  print(msg) // Set breakpoint here
}

struct Struct {
  static func sayStatic(_ msg: String) async {
    print("Set breakpoint here")
    await sayHello()
    print(msg) // Set breakpoint here
  }
}

@main struct Main {
  static func main() async {
    let closure = { msg in
      print("Set breakpoint here")
      await sayHello()
      print(msg) // Set breakpoint here
    }

    await sayBasic("basic world")
    await sayGeneric("generic world")
    await Struct.sayStatic("static world")
    await closure("closure world")
  }
}
