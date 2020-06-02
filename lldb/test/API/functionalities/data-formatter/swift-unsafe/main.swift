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
  // UnsafeBufferPointer
  let structArray = [ IntPair(1), IntPair(-2), IntPair(3) ]
  structArray.withUnsafeBufferPointer {
    let buf = $0
    print("break here ...")
    //% self.expect("frame variable -d run-target buf",
    //%            patterns=[
    //%            '\(UnsafeBufferPointer<(.*)\.IntPair>\) buf = 3 values \(0[xX][0-9a-fA-F]+\) {',
    //%            '\[0\] = \(original = 1, opposite = -1\)',
    //%            '\[1\] = \(original = -2, opposite = 2\)',
    //%            '\[2\] = \(original = 3, opposite = -3\)',
    //%            ])
  }
  
  // UnsafeMutableBufferPointer
  var enumArray = [ Toggle.Off ]
  enumArray.withUnsafeMutableBufferPointer {
    let mutbuf = $0
    print("... here ...")
    //% self.expect("frame variable -d run-target mutbuf",
    //%            patterns=[
    //%            '\(UnsafeMutableBufferPointer<(.*)\.Toggle>\) mutbuf = 1 value \(0[xX][0-9a-fA-F]+\) {',
    //%              '\[0\] = Off',
    //%            ])
    mutbuf[0] = Toggle.On
    print("... and here!")
    //% self.expect("frame variable -d run-target mutbuf",
    //%            patterns=[
    //%            '\(UnsafeMutableBufferPointer<(.*)\.Toggle>\) mutbuf = 1 value \(0[xX][0-9a-fA-F]+\) {',
    //%              '\[0\] = On',
    //%            ])
  }
  
  let colors = [ColorCode.RGB(155,219,255), ColorCode.Hex(0x4545ff)]
  
  colors.withUnsafeBufferPointer {
    let buf = $0
    //% self.expect("frame variable -d run-target buf",
    //%            patterns=[
    //%            '\(UnsafeBufferPointer<(.*)\.ColorCode>\) buf = 2 values \(0[xX][0-9a-fA-F]+\) {',
    //%            '\[0\] = RGB {',
    //%            'RGB = \(0 = 155, 1 = 219, 2 = 255\)',
    //%            '\[1\] = Hex \(Hex = 4539903\)',
    //%            ])
  }
  
  var flyingObjects : [Flyable] = [ Bird(), Plane() ]
  
  flyingObjects.withUnsafeMutableBufferPointer {
    let mutbuf = $0
    //% self.expect("frame variable -d run-target mutbuf",
    //%            patterns=[
    //%            '\(UnsafeMutableBufferPointer<(.*)\.Flyable>\) mutbuf = 2 values \(0[xX][0-9a-fA-F]+\) {',
    //%              '\[0\] = \(fly = "🦅"\)',
    //%              '\[1\] = \(fly = "🛩"\)',
    //%            ])
    struct UFO: Flyable {
      var fly: String = "🛸"
    }
    
    mutbuf[1] = UFO()
    //% self.expect("frame variable -d run-target mutbuf",
    //%            patterns=[
    //%            '\(UnsafeMutableBufferPointer<(.*)\.Flyable>\) mutbuf = 2 values \(0[xX][0-9a-fA-F]+\) {',
    //%              '\[0\] = \(fly = "🦅"\)',
    //%              '\[1\] = \(fly = "🛸"\)',
    //%            ])
  }
  
  let numbers = [ Number(number: 42), Number(number: 3.14)]
  
  numbers.withUnsafeBufferPointer {
    let buf = $0
    //% self.expect("frame variable -d run-target buf",
    //%            patterns=[
    //%            '\(UnsafeBufferPointer<(.*)\.Number<Double>>\) buf = 2 values \(0[xX][0-9a-fA-F]+\) {',
    //%            '\[0\] = 0[xX][0-9a-fA-F]+ \(number_value = 42\)',
    //%            '\[1\] = 0[xX][0-9a-fA-F]+ \(number_value = 3.14[0-9]*\)',
    //%            ])
  }
  
}

main()
