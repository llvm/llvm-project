import Foundation

@objc public class Foo : NSObject {}

extension Foo {
  public override var debugDescription: String { return "Foo from Swift!" }
}
