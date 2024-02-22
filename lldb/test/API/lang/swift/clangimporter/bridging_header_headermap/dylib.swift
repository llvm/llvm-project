import Foundation
func use<T>(_ t: T) {}

public class C<Foo> {

  public required init(_ f: Foo) { foo = f }
  var foo : Foo

  private struct Something {
    var txt = "hello"
  }

  public func f() {
    var a = Something()
    use(a) // break here
  }
}
