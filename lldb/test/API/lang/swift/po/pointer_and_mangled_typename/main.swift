struct Struct {
    var s = "Po"
    var n = 2025
}

class Class {
    var s = "Po"
    var n = 2025
}

enum Enum {
    case zero
    case pair(String, Int)
}

struct GenericStruct<T> {
    var s: T
    var n = 2025
}

class GenericClass<T> {
    var s: T
    var n = 2025
    init(s: T) { self.s = s }
}

enum GenericEnum<T> {
    case zero
    case pair(T, Int)
}

struct DescribedStruct: CustomStringConvertible {
    var s = "Po"
    var n = 2025
    var description: String { "DescribedStruct" }
}

class DescribedClass: CustomStringConvertible {
    var s = "Po"
    var n = 2025
    var description: String { "DescribedClass" }
}

enum DescribedEnum: CustomStringConvertible {
    case zero
    case pair(String, Int)
    var description: String { "DescribedEnum" }
}

@main struct Entry {
    static func main() {
        do {
            var value = 2025
            print("break int")
        }
        do {
            var value = "Po"
            print("break string")
        }
        do {
            let value = Struct()
            print("break struct")
        }
        do {
            let value = Class()
            print("break class")
        }
        do {
            let value = Enum.pair("Po", 50)
            print("break enum")
        }
        do {
            let value = GenericStruct(s: "Po")
            print("break generic struct")
        }
        do {
            let value = GenericClass(s: "Po")
            print("break generic class")
        }
        do {
            let value = GenericEnum.pair("Po", 50)
            print("break generic enum")
        }
        do {
            let value = DescribedStruct()
            print("break described struct")
        }
        do {
            let value = DescribedClass()
            print("break described class")
        }
        do {
            let value = DescribedEnum.pair("Po", 50)
            print("break described enum")
        }
    }
}
