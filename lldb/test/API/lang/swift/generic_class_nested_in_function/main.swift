// This test was created to ensure that the changes in in: apple/swift#61819 did not break behaviour.

func a() {
  class B<T>{
    func f() {
      print(1) // break here
    }
  }
  B<Int>().f()
}
a()
