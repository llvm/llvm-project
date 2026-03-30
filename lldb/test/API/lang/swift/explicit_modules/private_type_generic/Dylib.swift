public protocol MyProtocol {}

private struct PrivateImpl: MyProtocol {
  var value: Int
}

public func makeInstance() -> some MyProtocol {
  return PrivateImpl(value: 42)
}
