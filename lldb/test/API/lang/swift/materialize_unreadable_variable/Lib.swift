import NoRefl

// A resilient struct with one field whose reflection metadata is missing.
// Laying out the record fails, so a value of this type cannot be read at all.
public struct Unreadable {
    public var opaque: Opaque
    public var tag: Int
    public init(tag: Int) {
        self.opaque = Opaque(x: 1)
        self.tag = tag
    }
}

// A fully reflectable sibling.
public struct Plain {
    public var value: Int = 7
    public init() {}
}
