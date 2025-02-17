import Dylib
typealias LocalAlias = Foo
let local = LocalAlias()
let foo = MyAlias()
let bar = MyGenericAlias<MyAlias>()
let baz = MyGenericAlias<LocalAlias>()
print("\(local), \(foo), \(bar), \(baz)") // break here
