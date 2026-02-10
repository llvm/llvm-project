class GenericBase<T> {
    let baseValue: T

    init(_ v: T) {
        baseValue = v
    }
}

class GenericDerived<T, U>: GenericBase<T> {
    let derivedValue: U

    init(_ t: T, _ u: U) {
        derivedValue = u
        super.init(t)
    }
}

class GenericTriple<T, U, V>: GenericDerived<T, U> {
    let tripleValue: V

    init(_ t: T, _ u: U, _ v: V) {
        tripleValue = v
        super.init(t, u)
    }
}

class NonGenericBase {
    let baseField: Int = 100
}

class GenericFromNonGeneric<T>: NonGenericBase {
    let genericField: T

    init(_ t: T) {
        genericField = t
        super.init()
    }
}

// Multiple generic parameters
class MultiGenericBase<A, B> {
    let a: A
    let b: B

    init(_ a: A, _ b: B) {
        self.a = a
        self.b = b
    }
}

class MultiGenericDerived<A, B, C>: MultiGenericBase<A, B> {
    let c: C

    init(_ a: A, _ b: B, _ c: C) {
        self.c = c
        super.init(a, b)
    }
}

func f() {
    // Deliberately don't instantiate the base class.

    let derived = GenericDerived("hello", 100)
    let triple = GenericTriple("hello", 100, true)

    let fromNonGeneric = GenericFromNonGeneric(42)
    let fromNonGenericDouble = GenericFromNonGeneric(3.14)

    let multiDerived = MultiGenericDerived(1, 2.0, true)

    let string = StaticString("break here")
    print(string) // break here
}

f()
