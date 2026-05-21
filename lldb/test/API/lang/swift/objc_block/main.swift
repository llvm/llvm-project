class Container {
    var any: Any
    init(_ any: Any) { self.any = any }
}

func f() {
    let voidContainer = Container({} as @convention(block) () -> Void)
    let intContainer = Container({ $0 + 1 } as @convention(block) (Int) -> Int)
    print("break here")
}

f()
