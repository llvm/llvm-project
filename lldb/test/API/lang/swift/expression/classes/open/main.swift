open class A {
  open func foo() -> Int { return 23 }
}

func f() {
  let a  = A()
  print("break here \(a)")
}

f()
