import Module

struct Value {
  var x: Int32 = 1
  var s = Struct()
}

func f() {
  var v = Value()
  v.s.field_64_1 = 100;
  v.s.field_32_1 = 200;
  v.s.field_32_2 = 300;
  v.s.field_64_2 = 400;
  print(v) // break here
}

f()
