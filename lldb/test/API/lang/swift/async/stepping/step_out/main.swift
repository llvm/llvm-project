func work() {
  print("working")
}

func ASYNC___0___() async -> Int {
  var myvar = 111;
  work()
  myvar += 1; // BREAK HERE
  return myvar
}

func ASYNC___1___() async -> Int {
  var blah = 333
  work()
  var result = await ASYNC___0___()
  work()
  return result + blah
}

func ASYNC___2___() async -> Int {
  work()
  var result1 = await ASYNC___1___()
  work()
  return result1
}

func ASYNC___3___() async -> Int {
  var result = await ASYNC___2___()
  work()
  return result
}

@main struct Main {
  static func main() async {
    let result = await ASYNC___3___()
    print(result)
  }
}
