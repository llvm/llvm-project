func use<T>(_ t : T) {}

func string_tuple<T, U>(_ t : (T, U)) {
  let (_, y) = t // here
  use(y)
}


let s = "patatino"
string_tuple((s, s))

// CHECK-NOT: warning
