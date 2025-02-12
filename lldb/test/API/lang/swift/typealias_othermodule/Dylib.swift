public protocol Definition<Payload> {
  associatedtype Payload
  static func getPayload() -> Payload
}


public protocol Invokable {
  func perform()
}

public enum Impl : Definition {
  public typealias Payload = Bool
  case withPayload(Payload)
  public static func getPayload() -> Payload { return true }
}

public enum GenericImpl<T> : Definition {
  public typealias Payload = T
  case withPayload(Payload)
  public static func getPayload() -> Payload {
    let val : Bool = true
    return unsafeBitCast(val, to: T.self)
  }
}

public func getDefinition() -> any Definition {
    return Impl.withPayload(true)
}

public func getGenericDefinition() -> any Definition {
    return GenericImpl<Bool>.withPayload(true)
}
