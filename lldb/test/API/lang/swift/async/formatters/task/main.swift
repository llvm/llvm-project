func f() async -> Int {
    withUnsafeCurrentTask { currentTask in
        if let currentTask {
            print("break for current task")
        }
    }
    return 30
}

@main struct Main {
    static func main() async {
        let task = Task {
            // Extend the task's lifetime, hopefully long enough for the breakpoint to hit.
            try await Task.sleep(for: .seconds(0.5))
            print("inside")
        }
        print("break for top-level task")

        async let number = f()
        print(await number)
    }
}
