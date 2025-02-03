func first() async {
    await second()
}

func second() async {
    try? await Task.sleep(for: .seconds(300))
}

@main struct Main {
    static func main() async {
        let task = Task {
            await first()
        }
        try? await Task.sleep(for: .seconds(0.01))
        print("break here")
    }
}
