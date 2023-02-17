
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

public func returnSwiftClass() -> SwiftClass {
  return SwiftClass()
}

public func returnSwiftSubclassAsClass() -> SwiftClass {
  return SwiftSubclass()
}

public func returnSwiftStruct() -> SwiftStruct {
  return SwiftStruct()
}
