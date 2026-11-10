class K {
  var a = 1
  var b = 2
  func answer() -> Int { return 77 }
  static func staticAnswer() -> Int { return 66 }
  var computed: Int {
    return 55 // break in getter
  }
}

struct TwoInt8 {
  var a: Int8 = 1
  var b: Int8 = 2
}

struct TwoInt {
  var a = 3
  var b = 4
}

struct BoolInt32 {
  var b = true
  var i: Int32 = 9
}

struct HasClass {
  var k = K()
}

struct Counter {
  var x = 5
  mutating func increment() -> Int {
    x += 1
    return x
  }
}

struct E: Error { var code = 16 }

func f_int() -> Int { return 42 }
func f_int8() -> Int8 { return -5 }
func f_uint32() -> UInt32 { return 4000000000 }
func f_bool() -> Bool { return true }
func f_float() -> Float { return 1.5 }
func f_double() -> Double { return 2.25 }
func f_class() -> K { return K() }
func f_string() -> String { return "hello" }
func f_long_string() -> String {
  return "a string that is longer than fifteen bytes"
}
func f_two_int8() -> TwoInt8 { return TwoInt8() }
func f_two_int() -> TwoInt { return TwoInt() }
func f_bool_int32() -> BoolInt32 { return BoolInt32() }
func f_has_class() -> HasClass { return HasClass() }
func f_closure() -> () -> Int { return { 12 } }
func f_throws() throws -> Int { return 14 }
func f_throwing() throws -> Int { throw E() }

func main() {
  let v_int = f_int()
  let v_int8 = f_int8()
  let v_uint32 = f_uint32()
  let v_bool = f_bool()
  let v_float = f_float()
  let v_double = f_double()
  let v_class = f_class()
  let v_string = f_string()
  let v_long_string = f_long_string()
  let v_two_int8 = f_two_int8()
  let v_two_int = f_two_int()
  let v_bool_int32 = f_bool_int32()
  let v_has_class = f_has_class()
  let v_closure = f_closure()
  let v_throws = try! f_throws()
  do {
    _ = try f_throwing()
  } catch let v_throwing {
    _ = v_throwing
  }
  let k = K()
  let v_answer = k.answer()
  let v_staticAnswer = K.staticAnswer()
  let v_computed = k.computed
  var counter = Counter()
  let v_increment = counter.increment()
  _ = (v_int, v_int8, v_uint32, v_bool, v_float, v_double, v_class, v_string,
       v_long_string, v_two_int8, v_two_int, v_bool_int32,
       v_has_class, v_closure, v_throws, v_answer, v_staticAnswer, v_computed,
       v_increment)
}

main()
