public protocol P {
  func action()
}

public struct S<T> : P, CustomDebugStringConvertible {
  let m_t : T
  public var debugDescription: String { return "S { m_t = \(m_t) }" }
  public init(_ t : T) { m_t = t }
  public func action() {
    // self cannot be resolved in the expression evaluator,
    // because there is no AST type for it.
    let visible = Visible()
    print("break here")
  }
}

private struct Visible {
  let n = 42
  public init() {}
}

public func sanity_check() {
  let visible = Visible()
  print("break here")
}
