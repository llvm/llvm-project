func test() {
    let f0_0: () -> Void = { }
    let f0_1: () -> Int = { 42 }
    let f0_2: () -> (Int, Double) = { (1, 2.0) }

    let f1_0: (Float) -> Void = { _ in }
    let f1_1: (Int) -> Double = { Double($0) }
    let f1_2: (Double) -> (Float, Int) = { (Float($0), 1) }

    let f2_0: (Int, Double) -> Void = { _, _ in }
    let f2_1: (Float, Int) -> Double = { Double($0) + Double($1) }
    let f2_2: (Int, Float) -> (Double, Int) = { (Double($0), Int($1)) }

    let string = StaticString("break here")
    print(string) // break here
}

test()
