actor Actor {
    var data: Int = 15

    func occupy() async {
        _ = readLine()
    }

    func work() async -> Int {
        let result = data
        data += 1
        return result
    }
}

func breakHere(_ a: Actor) {}

@main struct Entry {
    static func main() async {
        let a = Actor()

        async let w: Void = a.occupy()
        // Provide time for the global concurrent executor to run this async
        // let, which enqueues a "blocking" job on the actor.
        try? await Task.sleep(for: .seconds(2))

        async let x = a.work()
        async let y = a.work()
        async let z = a.work()
        // Provide time for the global concurrent executor to kick off of these
        // async let tasks, which in turn enqueue jobs on the busy actor.
        try? await Task.sleep(for: .seconds(2))

        breakHere(a)
        await print(w, x, y, z)
    }
}
