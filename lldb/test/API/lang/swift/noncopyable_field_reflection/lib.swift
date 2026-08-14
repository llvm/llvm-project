// A plain, fully reflectable resilient struct.
public struct Plain {
    public var value: Int = 7
    public init() {}
}

// A noncopyable type stored in a resilient class.
public struct NC: ~Copyable {}

public class Holder {
    // Unresolvable reflection type ref.
    let field = NC()
    // A plain copyable sibling. Ideally it remains inspectable even though the
    // noncopyable field above cannot be resolved.
    public var tag: Int = 42
    public init() {}
}
