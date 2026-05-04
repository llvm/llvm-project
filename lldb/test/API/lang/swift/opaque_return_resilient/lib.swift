public protocol Producer {
    var name: String { get }
}

public class Creator: Producer {
    public let name: String
    public init(name: String) { self.name = name }
}

open class Manager {
    public init() {}

    public var producer: some Producer {
        return Creator(name: "alpha")
    }
}
