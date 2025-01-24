open class MyClass<A, B> {}
public typealias LocalAlias = Bool
let anchor : LocalAlias = true
public typealias ClassAlias = MyClass<LocalAlias, Bool>
class Repro {
  let field: ClassAlias?
  init(cls: ClassAlias?) {
    self.field = cls // break here
  }
}

Repro(cls: ClassAlias())
