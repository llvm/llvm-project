public enum Result<Value> {
  case success(Value)
  case failure(Error)
}

extension Result {
  public func map<U>(_ transform: (Value) -> U) -> Result<U> {
    switch self {
    case .success(let value):
      return .success(transform(value))
    case .failure(let error):
      return .failure(error)
    }
  }
}

func use<T>(_ t : T) {
}

public class SomeClass {
  public let s = "hello"
}

extension Result where Value : SomeClass {
  public func f() -> Self {
    use(self) // break one
    return map({ $0 })
  }  
}

extension Result {
  public func g() {
    use(self) // break two
  }  
}

let x : Result<SomeClass> = .success(SomeClass())
let y : Result<(Int64, Int64, Int64, Int64)> = .success((1, 2, 3, 4))
x.f()
y.g()
