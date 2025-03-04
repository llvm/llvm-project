@main struct Main {
    static func main() async {
        await withTaskGroup { group in
            for _ in 0..<3 {
                group.addTask {
                    try? await Task.sleep(for: .seconds(300))
                }
            }

            print("break here")
            for await _ in group {}
        }
    }
}
