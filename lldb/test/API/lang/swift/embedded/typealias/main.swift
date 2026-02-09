typealias MyInt = Int
typealias MyDouble = Double
typealias MyBool = Bool

typealias IntPair = (Int, Int)
typealias MixedTuple = (Int, Double, Bool)

typealias AliasOfMyInt = MyInt

struct SimpleStruct {
    var x: MyInt
}

struct MultiFieldStruct {
    var i: MyInt
    var d: MyDouble
    var b: MyBool
}

struct OptionalStruct {
    var maybeInt: MyInt?
}

struct TupleStruct {
    var pair: IntPair
    var mixed: MixedTuple
}

struct NestedAliasStruct {
    var nested: AliasOfMyInt
}

struct GenericWithAlias<T> {
    var value: T
    var aliased: MyInt
}

final class ClassWithAlias {
    var x: MyInt
    init(x: MyInt) { self.x = x }
}

struct Outer {
    typealias InnerInt = Int
    struct Inner {
        var x: InnerInt
    }
    var inner: Inner
}

func test() {
    let simple = SimpleStruct(x: 42)

    let multi = MultiFieldStruct(i: 1, d: 2.5, b: true)

    let optionalNil = OptionalStruct(maybeInt: nil)
    let optionalSome = OptionalStruct(maybeInt: 99)

    let tupleStruct = TupleStruct(pair: (10, 20), mixed: (1, 2.5, false))

    let nestedAlias = NestedAliasStruct(nested: 123)

    let genericAlias = GenericWithAlias<Int>(value: 314, aliased: 100)

    let classAlias = ClassWithAlias(x: 500)

    let outer = Outer(inner: Outer.Inner(x: 777))

    print("break here") // break here
}

test()
