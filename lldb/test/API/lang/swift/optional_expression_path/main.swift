struct Leaf { var a: Int; var b: Int }
struct Holder { var opt: Leaf?; var n: Int }

func use(_ leaf: Leaf?) -> Int {
    if let leaf { return leaf.a + leaf.b }
    return 0
}

func f() {
    let holder = Holder(opt: Leaf(a: 7, b: 8), n: 5)
    let plain: Leaf? = Leaf(a: 1, b: 2)
    let scalar: Int? = 42
    let nested: Leaf?? = Leaf(a: 3, b: 4)
    let empty: Leaf? = nil
    print("break here")
    _ = use(holder.opt) + use(plain) + (scalar ?? 0) + use(nested ?? nil)
        + use(empty)
}

f()
