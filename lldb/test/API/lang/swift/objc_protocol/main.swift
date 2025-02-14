class SwiftClass: NSObject, ObjcProtocol {
    let a = 42
    let b = 938
} 

func foo(v: (any ObjcProtocol)) {
    print(v) // break here
}

func f() {
    let swiftClass = SwiftClass()
    foo(v: swiftClass)
    let objcClass = ObjcClass()!
    foo(v: objcClass)
}
f()

