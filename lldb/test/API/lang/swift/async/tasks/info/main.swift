@main struct Main {
    static func main() async {
        withUnsafeCurrentTask { task in
            if let task {
                print("break here")
            }
        }
    }
}
