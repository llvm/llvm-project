@main struct Main {
    static func main() async {
        let task = Task {
            try? await Task.sleep(for: .seconds(100))
        }
        try? await Task.sleep(for: .seconds(0.5))
        print("break here")
    }
}
