import ModWithSuper

class Class<T>: SuperClass<T> {
    var v = 42

    override init(_ t: T) {
        super.init(t)
    }

    func f() -> Int {
        let abc = v
        return v
    }
}

public func entry() {
    let c = Class(true)
    print("break here")
}

