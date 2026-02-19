struct S {
  let x: Int
  let y: String
}

class C {
  init(_ v: Int) { self.v = v }
  let v: Int
}

struct LargePayload {
  let a: Int
  let b: Int
  let c: Int
  let d: String
}

// Single-case indirect enum with a struct payload.
indirect enum SingleStruct {
  case value(S)
}

// Single-case indirect enum with a class payload.
indirect enum SingleClass {
  case value(C)
}

// Single-case indirect enum with a tuple payload.
indirect enum SingleTuple {
  case value(Int, String)
}

// Single-case indirect enum with a large struct payload.
indirect enum SingleLargePayload {
  case value(LargePayload)
}

// Single-case indirect enum with a scalar payload.
indirect enum SingleInt {
  case value(Int)
}

// Recursive single-case indirect enum (linked list).
indirect enum List<T> {
  case node(T, List<T>?)
}

func main() {
  let single_struct = SingleStruct.value(S(x: 42, y: "hello"))
  let single_class = SingleClass.value(C(100))
  let single_tuple = SingleTuple.value(7, "world")
  let single_large = SingleLargePayload.value(
      LargePayload(a: 1, b: 2, c: 3, d: "big"))
  let single_int = SingleInt.value(99)
  let list = List<Int>.node(10, List<Int>.node(20, List<Int>.node(30, nil)))
  print("break here")
}

main()
