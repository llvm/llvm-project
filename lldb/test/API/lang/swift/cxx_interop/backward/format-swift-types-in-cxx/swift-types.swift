
public class SwiftClass {
  var field = 42
  var arr = ["An", "array", "of", "strings"]
}

public class SwiftSubclass: SwiftClass {
  var extraField = "this is an extra subclass field"
}

public struct SwiftStruct {
  var str = "Hello this is a big string"
  var boolean = true
}

@frozen
public struct GenericStructPair<T, T2> {
    var x: T
    var y: T2

    init(x: T, y: T2) {
        self.x = x
        self.y = y
    }
}

@frozen public enum GenericEnum<T> {
    case none
    case some(T)
}

public func returnSwiftClass() -> SwiftClass {
  return SwiftClass()
}

public func returnSwiftSubclass() -> SwiftSubclass {
  return SwiftSubclass()
}

public func returnSwiftStruct() -> SwiftStruct {
  return SwiftStruct()
}

public func returnStructPair<T, U>(t: T, u: U) -> GenericStructPair<T, U> {
  return GenericStructPair<T, U>(x: t, y: u)
}

public func returnGenericEnum<T>(t: T) -> GenericEnum<T> {
  return .some(t)
}

public typealias MyAliasedClass = SwiftClass;

public func returnAliasedClass() -> MyAliasedClass {
  return SwiftClass()
}
