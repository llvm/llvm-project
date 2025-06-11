protocol RandomNumberGenerator {
    func random(in range: ClosedRange<Int>) async -> Int
}

class SlowRandomNumberGenerator: RandomNumberGenerator {
    func random(in range: ClosedRange<Int>) async -> Int {
        try? await Task.sleep(for: .milliseconds(500))
        return Int.random(in: range)
    }
}

func doMath<RNG: RandomNumberGenerator>(with rng: RNG) async {
    print("break here")
    let y = await rng.random(in: 101...200)
    print("Y is \(y)")
}

let rng = SlowRandomNumberGenerator()
await doMath(with: rng)
