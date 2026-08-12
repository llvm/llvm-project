// A class-bound protocol. Inside its extension methods the static type of
// `self` is the generic archetype `τ_0_0`, not a concrete class.
protocol P: AnyObject {
    func payload() -> Int
}

class C: P {
    let number = 42
    func payload() -> Int { number }
}

extension P {
    func useSelf() {
        return // break here
    }
}

let c: P = C()
c.useSelf()
