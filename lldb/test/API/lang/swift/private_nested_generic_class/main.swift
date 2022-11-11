public class A<T> {
  private class B<U> {
    func f() {
        print(1) // break here
    }
  }
  func g() {
    B<String>().f()
  }
}

A<Int>().g()
