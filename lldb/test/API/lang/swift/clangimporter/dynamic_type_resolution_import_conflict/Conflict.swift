import Bar
public class C {
  public var i = 123
  public let f = FooFoo(i: 42)
}

public var foofoo = C()

@_silgen_name("init_conflict") public func init_conflict() {
  foofoo = C()
}
