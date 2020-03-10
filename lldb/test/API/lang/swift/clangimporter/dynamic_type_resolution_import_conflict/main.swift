import Bar
import Dylib

public class FromMainModule : LibraryProtocol {
  let i = 1
}

init_conflict()
let foobar = FooBar(j: 42)
Foo(FromMainModule()) // break here
