public struct S {
  public init() {}
  public let pub = 1
  private let priv = 2
  fileprivate let filepriv = 3

  private let s_priv = SPriv()
  fileprivate let s_filepriv = SFilePriv()
}

private struct SPriv {
  let i = 2
}

fileprivate struct SFilePriv {
  let i = 3
}

open class Base {
  public let x : Int = 42
  public init() {}
}

public enum NoPayload {
  case first
  case second
}

public enum WithPayload {
  case empty
  case with(i: Int)
}

public protocol P {}
