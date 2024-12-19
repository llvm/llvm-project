@_originallyDefinedIn(
     module: "Other", iOS 2.0, macOS 2.0, tvOS 2.0, watchOS 2.0)
@available(iOS 1.0, macOS 1.0, tvOS 1.0, watchOS 1.0, *)
public struct A {
    let i = 10
}

@_originallyDefinedIn(
     module: "Other", iOS 2.0, macOS 2.0, tvOS 2.0, watchOS 2.0)
@available(iOS 1.0, macOS 1.0, tvOS 1.0, watchOS 1.0, *)
public struct B {
    let i = 20
}

typealias Alias = B

@_originallyDefinedIn(
     module: "Other", iOS 2.0, macOS 2.0, tvOS 2.0, watchOS 2.0)
@available(iOS 1.0, macOS 1.0, tvOS 1.0, watchOS 1.0, *)
public enum C {
    public struct D {
        let i = 30
    }
}

@_originallyDefinedIn(
     module: "Other", iOS 2.0, macOS 2.0, tvOS 2.0, watchOS 2.0)
@available(iOS 1.0, macOS 1.0, tvOS 1.0, watchOS 1.0, *)
public enum E<T> {
    case t(T)
    case empty
}

@_originallyDefinedIn(
     module: "Other", iOS 2.0, macOS 2.0, tvOS 2.0, watchOS 2.0)
@available(iOS 1.0, macOS 1.0, tvOS 1.0, watchOS 1.0, *)
public class F {
    let i = 40
}

@_originallyDefinedIn(
     module: "Other", iOS 2.0, macOS 2.0, tvOS 2.0, watchOS 2.0)
@available(iOS 1.0, macOS 1.0, tvOS 1.0, watchOS 1.0, *)
public enum G {
    case i(Int)
    case empty
}

public struct Prop {
    let i = 50
}

@_originallyDefinedIn(
     module: "Other", iOS 2.0, macOS 2.0, tvOS 2.0, watchOS 2.0)
@available(iOS 1.0, macOS 1.0, tvOS 1.0, watchOS 1.0, *)
public struct Prop2 {
    let i = 70
}

@_originallyDefinedIn(
     module: "Other", iOS 2.0, macOS 2.0, tvOS 2.0, watchOS 2.0)
@available(iOS 1.0, macOS 1.0, tvOS 1.0, watchOS 1.0, *)
public struct Pair<T, U> {
    let t: T
    let u: U
}

@_originallyDefinedIn(
     module: "Other", iOS 2.0, macOS 2.0, tvOS 2.0, watchOS 2.0)
@available(iOS 1.0, macOS 1.0, tvOS 1.0, watchOS 1.0, *)
public class ClassPair<T, U> {
    let t: T
    let u: U
    init(t: T, u: U) {
        self.t = t
        self.u = u
    }
}

func generic<T>(_ t: T) {
    print("break for generic") 
}

func f() {
    let a = A()
    let b = Alias()
    let d = C.D()
    let e = E<Prop>.t(Prop())
    let f = F()
    let g = G.i(60)
    let h = ClassPair(t: Prop(), u: Prop2())
    let i = (A(), F(), Prop())
    let complex = Pair(t: E.t(Pair(t: Prop2(), u: C.D())), u: E.t(Prop()))
    print("break here")
}

func g() {
    generic(A())
    generic(Alias())
    generic(C.D())
    generic(E<Prop>.t(Prop()))
    generic(F())
    generic(G.i(60))
    generic(ClassPair(t: Prop(), u: Prop2()))
    generic((A(), F(), Prop()))
    generic(Pair(t: E.t(Pair(t: Prop2(), u: C.D())), u: E.t(Prop())))
}

f()
g()
