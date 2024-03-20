import Macro

func testStringify(a: Int, b: Int) {
  print("break here")
  let s = #stringify(a / b)
  print(s.1)
}

testStringify(a: 23, b: 0)
