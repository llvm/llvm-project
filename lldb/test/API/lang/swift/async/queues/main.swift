func async_foo() async -> Int {
  var myvar = 111; // BREAK HERE
  return myvar
}

@main struct Main {
  static func main() async {
    let result = await async_foo()
    print(result)
  }
}
