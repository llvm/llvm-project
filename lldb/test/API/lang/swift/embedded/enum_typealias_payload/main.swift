typealias MyInt = Int
typealias MyDouble = Double

enum AliasedPayloadEnum {
    case intCase(MyInt)
    case doubleCase(MyDouble)
    case empty
}

enum NestedAliasedPayloadEnum {
    typealias MyNestedInt = Int
    typealias MyNestedDouble = Double
    case first(MyNestedInt)
    case second(MyNestedDouble)
}

enum GenericAliasedPayloadEnum<T> {
    typealias Value = T
    case payload(Value)
    case empty
}

func f() {
    let intCase = AliasedPayloadEnum.intCase(42)
    let doubleCase = AliasedPayloadEnum.doubleCase(3.14)
    let empty = AliasedPayloadEnum.empty
    let first = NestedAliasedPayloadEnum.first(100)
    let second = NestedAliasedPayloadEnum.second(2.5)
    let genericInt = GenericAliasedPayloadEnum<Int>.payload(99)
    let genericDouble = GenericAliasedPayloadEnum<Double>.payload(1.5)
    let genericEmpty = GenericAliasedPayloadEnum<Int>.empty
    let string = StaticString("break here")
    print(string)
}
f()
