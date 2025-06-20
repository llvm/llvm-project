import Foundation
import Base

private class Bar : Base {
}

extension Bar {
  override var debugDescription : String { "Hello from Swift" }
}

@objc public class Foo : NSObject {
  @objc public func getBase() -> Base? {
    return Bar()
  }
}
