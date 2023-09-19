
func f<T>(t: T) {
  print(1) // break here for free function
}
f(t: "This is a string")
f(t: "This is another string")
f(t: true)
f(t: 5)

class MyClass<T> {
  func f() {
    print(1) // break here for class
  }
}
MyClass<String>().f()
MyClass<String>().f()
MyClass<Bool>().f()
MyClass<Int>().f()
