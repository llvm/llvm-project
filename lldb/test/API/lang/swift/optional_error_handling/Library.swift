public struct WithOpaqueType {
  public init() {}
  let opaqueSome : FromC? = FromC(i: 23)
  let opaqueNone : FromC? = nil
}
