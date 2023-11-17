protocol P {}

class C: P {
}

struct S<T> {
  let t: T
}
func f() -> some P {
  let p: some P = C()
  return p
}

let s = S(t: f())
print(s) // break here



