@_implementationOnly import Invisible

public func getObject() -> Any {
  return InvisibleStruct()
}

private class Conforming : InvisibleProtocol {
  let name = "conforming"
}

public func getConformingObject() -> Any {
  let proto : InvisibleProtocol = Conforming()
  return proto
}
