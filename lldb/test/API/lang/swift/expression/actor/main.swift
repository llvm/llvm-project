actor A {
    let i = 42
    let s = "Hello"
    static func foo() {
        print("break here for static")
    }

    func bar() {
        print("break here for func")
    }
}

A.foo()
await A().bar()
