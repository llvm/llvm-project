public class SwiftClass {
  let str = "Hello from the Swift class!"
}

@_expose(Cxx)
public func createArray() -> [SwiftClass] {
    return [SwiftClass(), SwiftClass()]
}

@_expose(Cxx)
public func createArrayOfInts() -> [CInt] {
    return [1, 2, 3, 4]
}

@_expose(Cxx)
public func createDict() -> [CInt : SwiftClass] {
    return [1: SwiftClass(), 4: SwiftClass()]
}


@_expose(Cxx)
public func createOptional() -> SwiftClass? {
  return SwiftClass()
}

@_expose(Cxx)
public func createOptionalPrimitive() -> Double? {
  return 4.2
}

@_expose(Cxx)
public func createString() -> String {
  return "Hello from Swift!"
}
