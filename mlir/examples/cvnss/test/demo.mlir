module {
  func.func @pack() -> i32 {
    %id = "cvnss.pack27"() {hi = 1 : i32, mid = 0 : i32, lo = -1 : i32} : () -> i32
    return %id : i32
  }
  func.func @unpack() -> (i32, i32, i32) {
    %a, %b, %c = "cvnss.unpack27"() {id = 21 : i32} : () -> (i32, i32, i32)
    return %a, %b, %c : i32, i32, i32
  }
}
