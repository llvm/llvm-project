import dynamic_lib

struct MyP : P {
  let i = 1
}

class C : Base {
    func f<T>(_ t: T, _ p : P) {
    let number = 1
    let array = [1, 2, 3]
    let string = "hello"
    let tuple = (0, 1)
    let strct = S()
    let generic = t
    let generic_tuple = (t, t)
    let word = 0._builtinWordValue
    let enum1 = NoPayload.second
    let enum2 = WithPayload.with(i:42)
    print("Set breakpoint here")
  }
}

let c = C()
c.f(42, MyP())
