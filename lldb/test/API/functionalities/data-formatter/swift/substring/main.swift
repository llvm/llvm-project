func main() {
    exerciseSmallString()
    exerciseString()
}

func exerciseSmallString() {
    exercise("abc")
}

func exerciseString() {
    exercise("abcdefghijklmnopqrstuvwxyz")
}

func exercise(_ string: String) {
    let substrings = allIndices(string).map { string[$0..<string.endIndex] }
    // break here
}

func allIndices<T: Collection>(_ collection: T) -> [T.Index] {
    return Array(collection.indices) + [collection.endIndex]
}

main()
