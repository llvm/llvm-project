enum MultiPayloadEnum {
    case single(Int)
    case pair(Int, Double)
    case triple(Int, Double, Bool)
}

enum SameSizePayloads {
    case intPair(Int, Int)
    case doublePair(Double, Double)
}

enum MixedPayloadEnum {
    case empty
    case oneArg(Int)
    case twoArgs(Int, Double)
    case threeArgs(Int, Double, Bool)
}

struct Small { let x: Int = 1 }
struct Medium { let x: Int = 1; let y: Int = 2 }
struct Large { let x: Int = 1; let y: Int = 2; let z: Int = 3 }

enum StructPayloadEnum {
    case small(Small)
    case medium(Medium)
    case large(Large)
}

func f() {
    // MultiPayloadEnum cases
    let multiSingle = MultiPayloadEnum.single(42)
    let multiPair = MultiPayloadEnum.pair(1, 2.5)
    let multiTriple = MultiPayloadEnum.triple(1, 2.5, true)

    // SameSizePayloads cases
    let intPair = SameSizePayloads.intPair(10, 20)
    let doublePair = SameSizePayloads.doublePair(1.5, 2.5)

    // MixedPayloadEnum cases
    let mixedEmpty = MixedPayloadEnum.empty
    let mixedOne = MixedPayloadEnum.oneArg(100)
    let mixedTwo = MixedPayloadEnum.twoArgs(100, 200.5)
    let mixedThree = MixedPayloadEnum.threeArgs(100, 200.5, false)

    // StructPayloadEnum cases
    let structSmall = StructPayloadEnum.small(Small())
    let structMedium = StructPayloadEnum.medium(Medium())
    let structLarge = StructPayloadEnum.large(Large())

    let string = StaticString("break here")
    print(string) // break here
}

f()
