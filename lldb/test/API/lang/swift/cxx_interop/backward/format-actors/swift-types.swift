@_expose(Cxx)
public final actor Actor {
  let str = "Hello"
}

@_expose(Cxx)
public func getActor() -> Actor {
  return Actor()
}
