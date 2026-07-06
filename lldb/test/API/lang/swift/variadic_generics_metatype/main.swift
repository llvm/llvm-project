public struct OuterStruct<each A> {
    public init() {}
}

func makeOuter<each B>() -> OuterStruct<repeat each B> {
    return OuterStruct<repeat each B>()
}

func f<each T>(_ types: repeat (each T).Type) {
    let x: OuterStruct<repeat each T> = makeOuter()
    print("break here")
    _ = x
}

func g<each T, each U>(_ pairs: repeat (each T, each U)) {
    let y: OuterStruct<repeat each U> = makeOuter()
    print("break here")
    _ = y
}

f(Int.self, String.self)
g((1, "a"), (2, "b"))
