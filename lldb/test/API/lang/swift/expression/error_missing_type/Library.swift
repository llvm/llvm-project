private enum E {
case WithString(String)
case OtherCase
}

public struct S {
  private var e : E { get { return .WithString("hidden") } }
}

public func getS() -> S { return S() }
