import Public

private struct InvisibleStruct {
  public var name = "The invisible man."
}

private class InvisibleClass {
  public var name = "The invisible class"
  public var someNumber = 42
}

public func privateDoIt()  {
  let structWrapper = StructWrapper(InvisibleStruct())
  structWrapper.foo()


  let classWrapper = ClassWrapper(InvisibleStruct())
  classWrapper.foo()

  let nonGeneric = NonGeneric()
  nonGeneric.foo()

  let twoGenericParameters = TwoGenericParameters(InvisibleStruct(), InvisibleClass())
  twoGenericParameters.foo()
}
