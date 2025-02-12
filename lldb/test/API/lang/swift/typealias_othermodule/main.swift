import Dylib

struct Invocation<Def : Definition> : Invokable {
  let payload : Def.Payload = Def.getPayload()
  func perform() {
    print("break here")
  }
}
func createInvocation<D : Definition>(_ def : D) -> any Invokable {
  return Invocation<D>()
}

createInvocation(getDefinition()).perform()
createInvocation(getGenericDefinition()).perform()
