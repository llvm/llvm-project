
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
public struct GenericPair<T, T2> {
    var x: T
    var y: T2

    init(x: T, y: T2) {
        self.x = x
        self.y = y
    }
}

public func returnSwiftClass() -> SwiftClass {
  return SwiftClass()
}

public func returnSwiftSubclassAsClass() -> SwiftClass {
  return SwiftSubclass()
}

public func returnSwiftStruct() -> SwiftStruct {
  return SwiftStruct()
}

public func returnPair<T, U>(t: T, u: U) -> GenericPair<T, U> {
  return GenericPair<T, U>(x: t, y: u)
}
