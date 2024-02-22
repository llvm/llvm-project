func swapTwoValues<T>(_ a: inout T, _ b: inout T) {
  let temporaryA = a //%self.expect('frame variable -d run -O -- a', substrs=['(UInt8) a = 97'])
  a = b
  b = temporaryA
}

func getASCIIUTF8() -> (UnsafeMutablePointer<UInt8>, dealloc: () -> ()) {
  let up = UnsafeMutablePointer<UInt8>.allocate(capacity: 100)
  up[0] = 0x61
  up[1] = 0x62
  swapTwoValues(&up[0], &up[1])
  return (up, { up.deallocate() })
}

let (_, dealloc) = getASCIIUTF8()
dealloc()
