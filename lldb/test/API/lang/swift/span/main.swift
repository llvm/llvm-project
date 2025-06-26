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
        let ints_span = ints.span
        let strings = ["six", "seven"]
        let strings_span = strings.span
        let things = [Thing(67)]
        let things_span = things.span
        print("break here")
    }
}
