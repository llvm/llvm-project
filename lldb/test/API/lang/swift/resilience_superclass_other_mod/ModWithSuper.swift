import Foundation

open class SuperClass<T>: NSObject {
    var someVar: T
    public init(_ someVar: T) {
        self.someVar = someVar
        super.init()
    }
}
