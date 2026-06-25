import Foundation

@objc(Builder) public class Builder: NSObject {
    @objc public class func makeArray() -> [Item] {
        return [Item(name: "hello"), Item(name: "world")]
    }
}
