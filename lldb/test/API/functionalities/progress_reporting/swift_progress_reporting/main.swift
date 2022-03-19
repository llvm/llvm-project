struct IntPair {
  var original: Int
  var opposite: Int

  init(_ value: Int) {
    self.original = value
    self.opposite = -value
  }
}

enum Toggle { case On; case Off }

enum ColorCode {
  case RGB(UInt8, UInt8, UInt8)
  case Hex(Int)
}

protocol Flyable {
  var fly : String { get }
}

struct Bird: Flyable {
  var fly: String = "🦅"
}

struct Plane: Flyable {
  var fly: String = "🛩"
}

class Number<T:Numeric> {
  var number_value : T

  init (number value : T) {
    number_value = value
  }
}

func main() {
  let structArray = [ IntPair(1), IntPair(-2), IntPair(3) ]
  var enumArray = [ Toggle.Off ]
  var colors = [ColorCode.RGB(155,219,255), ColorCode.Hex(0x4545ff)]
  var flyingObjects : [Flyable] = [ Bird(), Plane() ]
  let numbers = [ Number(number: 42), Number(number: 3.14)]
  let bytes = [UInt8](0...255)
  var bits : [UInt8] = [0,1]
  print("break here")
}

main()
