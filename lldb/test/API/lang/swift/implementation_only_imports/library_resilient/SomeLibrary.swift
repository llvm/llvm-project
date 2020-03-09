@_implementationOnly import SomeLibraryCore

public struct ContainsTwoInts {
  internal var wrapped: TwoInts
  public var other: Int

  public init(_ value: Int) {
    wrapped = .init(2, 3)
    other = value
  }
}
