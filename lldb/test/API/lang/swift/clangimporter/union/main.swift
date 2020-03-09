import U

func use<T>(_ t : T) {}

let i = IntDoubleUnion(i: 42)
let d = IntDoubleUnion(d: 23)
use((i, d)) // break here
