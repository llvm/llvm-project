protocol B: AnyObject {
    var name: String { get }
}

class ConcreteB: B {
    let name = "ConcreteB"
    let value = 42
}

class SubConcreteB: ConcreteB {
    let extra = 100
}

class A<SomeB: B> {
    let member: SomeB
    let id = 1

    init(member: SomeB) {
        self.member = member
    }
}

struct Statics {
    static let globalB: B = SubConcreteB()
    static let globalA = A(member: SubConcreteB())
    static let globalAWithProtocol: A<ConcreteB> = A(member: SubConcreteB())
    static let globalArrayOfB: [B] = [ConcreteB(), SubConcreteB()]
    static let globalArrayOfA = [A(member: SubConcreteB()), A(member: SubConcreteB())]
}

func f() {
    let b: B = SubConcreteB()
    let a = A(member: SubConcreteB())
    let aWithProtocol: A<ConcreteB> = A(member: SubConcreteB())
    let arrayOfB: [B] = [ConcreteB(), SubConcreteB()]
    let arrayOfA = [A(member: SubConcreteB()), A(member: SubConcreteB())]
    let _ = Statics.globalB
    let _ = Statics.globalA
    let _ = Statics.globalAWithProtocol
    let _ = Statics.globalArrayOfB
    let _ = Statics.globalArrayOfA
    print("break here")
}

f()
