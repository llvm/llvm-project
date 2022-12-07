
public struct StructWrapper<T> {
  let t: T

  public init(_ t: T) {
    self.t = t
  }
  public func foo() {
    print(self) // break here for struct
  }
}

public class ClassWrapper<T> {
  let t: T

  public init(_ t: T) {
    self.t = t
  }
  public func foo() {
    print(self) // break here for class
  }
}


public class NonGeneric {
  public init() {}

  public func foo() {
    print(self) // break here for non-generic
  }
}

public class TwoGenericParameters<T, U> {
  let t: T
  let u: U

  public init(_ t: T, _ u: U) {
    self.t = t
    self.u = u
  }

  public func foo() {
    print(self) // break here for two generic parameters
  }
}
