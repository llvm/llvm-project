func main() {
  var zero : String = "zero"
  withUnsafeMutablePointer(to: &zero) {
    $0.withMemoryRebound(to: UInt64.self, capacity: 2) { raw in
      raw[0] = 0
      raw[1] = 0
    }
  }
  var random : String = "random"
  withUnsafeMutablePointer(to: &random) {
    $0.withMemoryRebound(to: UInt64.self, capacity: 2) { raw in
      raw[0] = 0xfefefefefefefefe
      raw[1] = 0xfefefefefefefefe
    }
  }
  print("break here")
}

main()
