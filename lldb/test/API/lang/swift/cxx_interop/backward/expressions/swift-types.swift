
public func swiftFunc() -> String{
  return "Inside a Swift function!"
}

public class SwiftClass {
  var field: Int 
  var arr: [String]
  public init() {
    field = 42
    arr = ["An", "array", "of", "strings"]
  }

  public func swiftMethod() -> String {
    return "Inside a Swift method!"
  }
  
  private var _desc = "This is a class with properties!"
  public var swiftProperty: String {
    get {
      return _desc
    }
  }

  public static func swiftStaticMethod() -> String {
    return "In a Swift static method!"
  }

  public func overrideableMethod() -> String {
    return "In the base class!"
  }
}

public class SwiftSubclass: SwiftClass {
  public override func overrideableMethod() -> String {
    return "In subclass!"
  }
}

