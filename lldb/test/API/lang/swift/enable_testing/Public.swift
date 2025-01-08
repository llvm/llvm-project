class SomeClass {
    let value = 42
}

class ClassWithProperty {
    private var v = SomeClass()

    func f() {
        print("break here")
    }
}

public func entry() {
    ClassWithProperty().f()
}
