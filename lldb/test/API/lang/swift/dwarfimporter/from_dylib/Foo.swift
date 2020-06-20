import CFoo

public struct WrappingFromDylib {
  let s : FromDylib = FromDylib(i: 23)
}

private let anchor = WrappingFromDylib()

public func foo() {}
