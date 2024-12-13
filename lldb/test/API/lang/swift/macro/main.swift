import Macro
// This test depends on the layout of the lines in this file.
func testStringify(a: Int, b: Int) {
  print("break here")
  let s = #stringify(a / b)
  #no_return(a)
  #no_return(b)
  #no_return(a / b)
  print(s.1)
}

testStringify(a: 23, b: 1)
