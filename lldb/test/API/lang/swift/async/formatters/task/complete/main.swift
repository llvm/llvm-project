@main struct Main {
    static func main() async {
        let task = Task { return 15 }
        _ = await task.value
        print("break here")
    }
}
