public struct Foo {
    let i = 23
    public init() {}
}
public struct Bar<T> {
    let i = 42
    public init() {}
}
public typealias MyAlias = Foo
public typealias MyGenericAlias<T> = Bar<T>
