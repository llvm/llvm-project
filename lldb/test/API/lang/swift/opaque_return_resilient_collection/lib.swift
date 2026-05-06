public struct Creator {
    public let name: String
    public init(name: String) { self.name = name }
}

open class Manager {
    private var items: [Creator] = [
        Creator(name: "one"),
        Creator(name: "two"),
    ]

    public init() {}

    public var creators: some Collection<Creator> {
        return items
    }
}
