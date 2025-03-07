protocol P: AnyObject {

}

class Super: P {
    let superField = 42
}

class Sub: Super {
    let subField = 100
}

func f() {
    let s: Super = Sub()
    let p: P = Sub()
    print("break here")
}

f()
