import Library

public class One {
  let j = 98
  init() {}
}

public class Generic2<T> {
  let t2: T
  public init(t: T) {
    self.t2 = t
  }

  func foo() {
    print("break for self")
  }
}


func main() {
  let two = a.One()
  let generic1 = Library.Generic<a.One>(t: two)
  let generic2 = a.Generic2<Library.Generic<a.One>>(t: generic1)
  let generic3 = Library.Generic<a.Generic2<Library.Generic<a.One>>>(t: generic2)
  generic2.foo()
  print("break here")
}

main()
