import Public

private struct InvisibleStruct {
  public var name = "The invisible struct."
}

private class InvisibleClass {
  public var name = "The invisible class."
  public var someNumber = 42
}

public func privateDoIt()  {
  let structWrapper = StructWrapper(InvisibleStruct())
  structWrapper.foo()


  let classWrapper = ClassWrapper(InvisibleStruct())
  classWrapper.foo()

  let twoGenericParameters = TwoGenericParameters(InvisibleClass(), InvisibleStruct())
  twoGenericParameters.foo()

  let threeGenericParameters = ThreeGenericParameters(InvisibleClass(), InvisibleStruct(), true)
  threeGenericParameters.foo()
  
  let fourGenericParameters = FourGenericParameters(InvisibleStruct(), 
                                                      InvisibleClass(), 
                                                      ["One", "two", "three"], 
                                                      482)
  fourGenericParameters.foo()

  let nonGeneric = NonGeneric()
  nonGeneric.foo()

  let nestedParameters = Nested.Parameters(InvisibleClass(), InvisibleStruct())
  nestedParameters.foo()
}
