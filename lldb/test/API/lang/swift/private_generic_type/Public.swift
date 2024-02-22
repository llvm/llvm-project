
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

public struct ThreeGenericParameters<T, U, V> {
  let t: T
  let u: U
  let v: V

  public init(_ t: T, _ u: U, _ v: V) {
    self.t = t
    self.u = u
    self.v = v
  }

  public func foo() {
    print(self) // break here for three generic parameters
  }
}

public struct FourGenericParameters<T, U, V, W> {
  let t: T
  let u: U
  let v: V
  let w: W

  public init(_ t: T, _ u: U, _ v: V, _ w: W) {
    self.t = t
    self.u = u
    self.v = v
    self.w = w
  }

  public func foo() {
    print(self) // break here for four generic parameters
  }
}

public struct Nested<T> {
  public struct Parameters<U>  {
    let t: T
    let u: U

    public init(_ t: T, _ u: U) {
      self.t = t
      self.u = u
    }

    public func foo() {
      print(self) // break here for nested generic parameters
    }
  }
}

