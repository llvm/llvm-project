func f() async -> Int {
    try? await Task.sleep(for: .seconds(300))
    return 30
}

@main struct Main {
    static func main() async {
        async let number = f()
        await print("break here \(number)")
    }
}
