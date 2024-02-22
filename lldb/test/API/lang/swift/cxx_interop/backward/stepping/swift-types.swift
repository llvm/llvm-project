
public func swiftFunc() {
  print("Inside a Swift function!")
}

public class SwiftClass {
  var field: Int 
  var arr: [String]
  public init() {
    field = 42
    arr = ["An", "array", "of", "strings"]
  }

  public func swiftMethod() {
    print("Inside a Swift method!")
  }
  
  private var _desc = "This is a class with properties!"
  public var swiftProperty: String {
    get {
      return _desc
    }
    set {
      _desc = newValue
    }
  }

  public static func swiftStaticMethod() {
    print("In a Swift static method!")
  }

  public func overrideableMethod() {
    print("In the base class!")
  }
}

public class SwiftSubclass: SwiftClass {
  public override func overrideableMethod() {
    print("In subclass!")
  }
}

public class SwiftStruct {
  var field: Int 
  var arr: [String]
  public init() {
    field = 42
    arr = ["An", "array", "of", "strings"]
  }

  public func swiftMethod() {
    print("Inside a Swift method!")
  }
  
  private var _desc = "This is a class with properties!"
  public var swiftProperty: String {
    get {
      return _desc
    }
    set {
      _desc = newValue
    }
  }

  public static func swiftStaticMethod() {
    print("In a Swift static method!")
  }
}

public func getString() -> String {
  return "A brand new string!";
}

