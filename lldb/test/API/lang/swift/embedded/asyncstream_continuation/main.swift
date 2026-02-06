import _Concurrency

struct DataItem {
    let id: Int
    let value: Int
}

@main struct Main {
    static func main() async {
        let stream = AsyncStream<DataItem> { continuation in
            print("break here") // break here
            continuation.yield(DataItem(id: 1, value: 100))
            continuation.finish()
        }

        for await item in stream {
            print("Received item: \(item.id)")
        }
    }
}
