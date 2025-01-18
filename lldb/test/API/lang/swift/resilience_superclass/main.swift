import Foundation

public class SuperClass<T>: NSObject {
    var someVar: T
    init(_ someVar: T) {
        self.someVar = someVar
        super.init()
    }
}

class Class<T>: SuperClass<T> {
    var v = 42

    override init(_ t: T) {
        super.init(t)
    }
}


open class OpenSuperClass<T>: NSObject {
    var someVar: T
    init(_ someVar: T) {
        self.someVar = someVar
        super.init()
    }
}

class InheritingOpenClass<T>: OpenSuperClass<T> {
    var v = 100

    override init(_ t: T) {
        super.init(t)
    }
}

func main() {
    let c = Class(true)
    let c2 = InheritingOpenClass(true)
    print("break here")
}

main()
