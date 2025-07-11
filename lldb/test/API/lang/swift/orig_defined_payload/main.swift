@frozen
@available(macOS 10.0, macCatalyst 13.0, iOS 2.0, watchOS 1.0, tvOS 9.0, *)
@_originallyDefinedIn(module: "SomeModule", macOS 10.0, macCatalyst 13.0, iOS 2.0, watchOS 1.0, tvOS 9.0)
public struct TheStruct {
    public let value: Double
    public init(_ v: Double ) {
        value = v
    }
}

enum TheEnum {
    case fixed(TheStruct)
    case flexible(TheStruct?, TheStruct, TheStruct?)
}

func f(c: [TheEnum]) {
    print(c) // break here
}

f(c: [.flexible(TheStruct(100), TheStruct(200), TheStruct(300))])
