func f1(_ ptr: UnsafePointer<UnsafePointer<Int>?>) {
    print("pointer: \(ptr)") // break here
}

func f2(_ ptr: UnsafePointer<Int?>) {
    print("pointer: \(ptr)") // break here
}

var ptr: UnsafePointer<Int>? = nil
f1(&ptr)
var n: Int? = nil
f2(&n)
