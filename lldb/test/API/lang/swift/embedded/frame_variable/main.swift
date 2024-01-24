struct A {
  let field = 4.2
}

struct B {
  let a = A()
  let b = 123456
}

let varB = B()
let tuple = (A(), B())
// Dummy statement to set breakpoint print can't be used in embedded Swift for now.
let dummy = A() // break here

