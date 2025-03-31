protocol P {}

class C: P {
    let i = 42
}

struct S<T> {
  let t: T
}
func f() -> some P {
  let p: some P = C()
  return p
}

let v = f()
let s = S(t: f())
print(s) // break here



