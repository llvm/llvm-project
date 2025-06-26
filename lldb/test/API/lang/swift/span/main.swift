struct Thing {
    var id: Int
    var odd: Bool

    init(_ n: Int) {
        id = n
        odd = n % 2 == 1
    }
}

@main struct Entry {
    static func main() {
        let ints = [6, 7]
        let strings = ["six", "seven"]
        let things = [Thing(67)]
        print("break here")
    }
}
