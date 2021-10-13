import dynamic_lib

protocol P {
    func f()
}

class C : P {
    func f() {
        let c = getS()
        print(c) // Set breakpoint here
    }
}

let c : P = C()
c.f()
