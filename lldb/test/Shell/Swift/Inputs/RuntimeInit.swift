func use<T>(_ t: T) {}

protocol P { var p: Int { get } }
class C: P { let p: Int = 42 }

let p2: P = C()
use(p2) // break here
