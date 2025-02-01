import Builtin

@frozen
public struct Vector<let Count: Int, Element: ~Copyable>: ~Copyable {
    private var storage: Builtin.FixedArray<Count, Element>

    public init(_ valueForIndex: (Int) -> Element) {
        storage = Builtin.emplace { rawPointer in
            let base = UnsafeMutablePointer<Element>(rawPointer)
            for i in 0..<Count {
                (base + i).initialize(to: valueForIndex(i))
            }
        }
    }

    public subscript(i: Int) -> Element {
        _read {
            assert(i >= 0 && i < Count)
            let rawPointer = Builtin.addressOfBorrow(self)
            let base = UnsafePointer<Element>(rawPointer)
            yield ((base + i).pointee)
        }

        _modify {
            assert(i >= 0 && i < Count)
            let rawPointer = Builtin.addressof(&self)
            let base = UnsafeMutablePointer<Element>(rawPointer)
            yield (&(base + i).pointee)
        }
    }
}
extension Vector: Copyable where Element: Copyable {
    public init(repeating value: Element) {
        self.init { _ in value }
    }
}
extension Vector: BitwiseCopyable where Element: BitwiseCopyable {}

func main() {
    var ints = Vector<4, Int>(repeating: 0)
    ints[0] = 0
    ints[1] = 1
    ints[2] = 2
    ints[3] = 3
    var bools = Vector<2, Bool>(repeating: false)
    bools[0] = false
    bools[1] = true

    struct S { let i : Int; let j : Int };
    var structs = Vector<2, S>(repeating: S(i: 1, j: 2))
    structs[0] = S(i: 1, j: 2)
    structs[1] = S(i: 3, j: 4)
    
    print("\(ints), \(bools), \(structs)")  // break here
}
main()
