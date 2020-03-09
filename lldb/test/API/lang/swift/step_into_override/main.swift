class Base {
  func foo (_ a: Int!) {
    print(a)
  }
}

class Sub : Base {
  func foo (_ a: Int) {
    print(a+1)
  }
}

let b : Base = Sub()
b.foo(3) // break here 
