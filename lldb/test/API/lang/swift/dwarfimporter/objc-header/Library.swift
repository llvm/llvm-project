import Foundation

/// This is an object exposed to Objective-C as "_Tt7Library10SwiftClassC".
@objc public class MangledSwiftClass : NSObject {
  @objc public func getString() -> NSString { return "Hello from Swift!" }
}

/// This is an object exposed to Objective-C as "RawNameSwiftClass".
@objc(RawNameSwiftClass) public class RawNameSwiftClass : NSObject {
  @objc public func getString() -> NSString { return "Hello from Swift!" }
}
