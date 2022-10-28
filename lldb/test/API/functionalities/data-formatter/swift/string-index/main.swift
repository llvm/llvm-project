import Foundation

func main() {
    exerciseNative()
    exerciseBridged()
}

func exerciseNative() {
    exercise("a👉🏼b")
}

func exerciseBridged() {
    exercise("a👉🏼b" as NSString as String)
}

func exercise(_ string: String) {
    let nativeIndices = allIndices(string)
    let unicodeScalarIndices = allIndices(string.unicodeScalars)
    let utf8Indices = allIndices(string.utf8)
    let utf16Indices = allIndices(string.utf16)
    // break here
}

func allIndices<T: Collection>(_ collection: T) -> [T.Index] {
    return Array(collection.indices) + [collection.endIndex]
}

main()
