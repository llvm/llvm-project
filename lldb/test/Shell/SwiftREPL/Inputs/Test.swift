public struct Bar {
    public init(baz: Int) { self.baz = baz }

    var baz : Int
}

struct Foo {
    init(bar: Int) { self.bar = bar }

    var bar : Int
}
