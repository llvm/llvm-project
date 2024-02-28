struct A {
  var field = 4

  func foo() -> Int {
    return field * field
  }
}


let a = A()
// Dummy statement to set breakpoint print can't be used in embedded Swift for now.
let dummy = A() // break here
let string = StaticString("Hello") 
print(string) 

