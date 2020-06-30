func use<T>(_ t: T) {}

public class Foo {
  public init() {}
  public func f() {
    let foo = 42
    use(foo) // break here
  }
}
