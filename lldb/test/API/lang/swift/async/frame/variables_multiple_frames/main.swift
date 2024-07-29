func work() {
  print("working") // breakpoint2
}

func ASYNC___1___() async -> Int {
  var myvar = 111;
  work()
  myvar += 1;
  return myvar
}

func ASYNC___2___() async -> Int {
  var myvar = 222;
  let result = await ASYNC___1___() // breakpoint1
  work() // breakpoint3
  myvar += result;
  return myvar
}

func ASYNC___3___() async -> Int {
  var myvar = 333;
  let result = await ASYNC___2___()
  work()
  myvar += result
  return myvar
}

func ASYNC___4___() async -> Int {
  var myvar = 444;
  let result = await ASYNC___3___()
  work()
  myvar += result
  return myvar
}

func ASYNC___5___() async -> Int {
  var myvar = 555;
  let result = await ASYNC___4___()
  work()
  myvar += result
  return myvar
}

@main struct Main {
  static func main() async {
    let result = await ASYNC___5___()
    print(result)
  }
}
