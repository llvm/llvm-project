class Object: CustomStringConvertible {
  var description: String {
    let address = unsafeBitCast(self, to: Int.self)
    let hexAddress = String(address, radix: 16)
    return "Object@0x\(hexAddress)"
  }
}

func main() {
    let object = Object()
    _ = object // break here
}

main()
