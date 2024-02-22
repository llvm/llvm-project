public class C {
    private var m: Int
    public func use_m() {
      print(m) // break here
    }
    public init() { m = 42 }
}

C().use_m()
