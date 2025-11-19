class HoldsNonNamespacedNestedStruct {
    struct NamespacedNestingStruct<T> {
        let nested: T
    }
    
}

func f() {
    let v = HoldsNonNamespacedNestedStruct.NamespacedNestingStruct<Int>(nested: 42)
    print(v) // break here
}

f()
