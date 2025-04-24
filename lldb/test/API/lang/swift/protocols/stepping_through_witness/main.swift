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
    let y = await rng.random(in: 101...200) // break here
    print("Y is \(y)")
}

let rng = SlowRandomNumberGenerator()
await doMath(with: rng)
