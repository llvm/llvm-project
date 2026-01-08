struct A {}

private protocol MyProtocolBase<T> {
    typealias MyAlias = A
    associatedtype T
}

private protocol MyProtocol : MyProtocolBase<Int> {
}

extension MyProtocol {
    func f1() {
        typealias Pair<T> = (T, T)
        let local : Pair<Int> = (1, 2)
        print("break here")
    }

    func f2(_ associated : MyAlias) {
        print("break here")
    }
}

struct B : MyProtocol {
}

func f() {
    typealias Pair<T> = (T, T)
    let local : Pair<Int> = (1, 2)
    print("break here")
}

f()
let b = B()
b.f1()
b.f2(A())
