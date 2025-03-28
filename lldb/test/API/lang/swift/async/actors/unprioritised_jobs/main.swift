import Foundation

actor Actor {
    var data: Int = 15

    func occupy() async {
        Thread.sleep(forTimeInterval: 100)
    }

    func work() async -> Int {
        let result = data
        data += 1
        return result
    }
}

@main struct Entry {
    static func main() async {
        let a = Actor()
        async let _ = a.occupy()
        async let _ = a.work()
        async let _ = a.work()
        async let _ = a.work()
        print("break here")
    }
}
