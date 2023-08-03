import Generic

private class Private{
  fileprivate typealias MyNumber = Int
  fileprivate init() { n = MyNumber(23) }
  let n : MyNumber
}

public func getObj() -> some P {
  return S(Private())
}
