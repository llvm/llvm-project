@main struct Main {
    static func main() async {
        await withTaskGroup { group in
            for _ in 0..<3 {
                group.addTask {
                    try? await Task.sleep(for: .seconds(1))
                }
            }

            print("break here TaskGroup")
            for await _ in group {}
        }

        try? await withThrowingTaskGroup { group in
            for _ in 0..<3 {
                group.addTask {
                    try await Task.sleep(for: .seconds(1))
                }
            }

            print("break here ThrowingTaskGroup")
            for try await _ in group {}
        }
    }
}
