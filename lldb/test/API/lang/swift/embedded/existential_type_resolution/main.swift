protocol P {
}

struct S: P {
    let structField = 111
}

class C: P {
    let classField = 222
}

class Sub: C {
    let subField = 333
}

enum E: P {
    case first
    case second(Int)
}

func f() {
    let pStruct: any P = S()
    let pClass: any P = C()
    let pSubclass: any P = Sub()
    // FIXME: rdar://168697959 (Debug info for enums is not generated, if there are no variables/parameter with the enum type in embedded swift)
    let bug = E.first
    let pEnumFirst: any P = E.first
    let pEnumSecond: any P = E.second(444)
    print("break here")
}

f()
