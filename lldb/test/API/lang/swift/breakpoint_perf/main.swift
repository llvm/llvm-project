
class C {
  func f() {
    {
      print("hello from a complicated context") // Set breakpoint here
    }()
  }
}

let c = C()
c.f()
