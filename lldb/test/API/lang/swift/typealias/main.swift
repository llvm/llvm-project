import Dylib
typealias LocalAlias = Foo
let local = LocalAlias()
let foo = MyAlias()
let bar = MyGenericAlias<MyAlias>()
print("\(local), \(foo), \(bar)") // break here
