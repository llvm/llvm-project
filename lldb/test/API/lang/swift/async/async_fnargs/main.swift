import Swift
func use<T>(_ t: T) {}

func sayHello() async {
  print("hello")
}

func sayGeneric<T>(_ msg: T) async {
  print("Set breakpoint here")
  await sayHello()
  print(msg) // And also here.
}

runAsyncAndBlock {
  await sayHello()
  await sayGeneric("world")
}
