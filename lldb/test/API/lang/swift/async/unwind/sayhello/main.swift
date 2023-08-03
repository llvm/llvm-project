import Swift

func sayolleH(str : String) async {
  var str = "in sayolleH"
  print("\(str) before calls")
  print(str.reversed()) // break here
}

func sayHello() async {
  var str = "in hello"
  print("\(str) before calls")
  await sayolleH(str:"hello")
  print("\(str) after calls")
}

func sayGeneric<T>(_ msg: T) async {
  var str = "in generic"
  print("\(str) before calls - arg \(msg)")
  await sayHello()
  print("\(str) after calls - arg \(msg)")
}

@main struct Main {
  static func main() async {
    await sayGeneric("world")
    await sayHello()
  }
}
