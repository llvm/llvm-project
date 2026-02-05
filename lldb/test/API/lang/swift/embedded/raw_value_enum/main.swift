// Test for raw value enum type resolution in embedded Swift
// Bug: Raw value enums fail with "unimplemented enum kind"

enum IntRawEnum: Int {
    case zero = 0
    case one = 1
    case two = 2
    case hundred = 100
}

enum Int8RawEnum: Int8 {
    case a = 1
    case b = 2
    case c = 3
}

enum UIntRawEnum: UInt {
    case x = 100
    case y = 200
    case z = 300
}

enum UInt8RawEnum: UInt8 {
    case first = 0
    case second = 1
    case third = 2
}

enum Int16RawEnum: Int16 {
    case neg = -100
    case zero = 0
    case pos = 100
}

func f() {
    let intZero = IntRawEnum.zero
    let intOne = IntRawEnum.one
    let intHundred = IntRawEnum.hundred

    let int8A = Int8RawEnum.a
    let int8B = Int8RawEnum.b

    let uintX = UIntRawEnum.x
    let uintY = UIntRawEnum.y

    let uint8First = UInt8RawEnum.first
    let uint8Second = UInt8RawEnum.second

    let int16Neg = Int16RawEnum.neg
    let int16Zero = Int16RawEnum.zero
    let int16Pos = Int16RawEnum.pos

    let string = StaticString("break here")
    print(string) // break here
}

f()
