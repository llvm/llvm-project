// This must be an @_implementationOnly import, otherwise importing A
// implicitly also imports A.B.
@_implementationOnly import A.B

struct s {
  fileprivate var priv : FromB { return FromB(i: 23) }
}

let a = s()
print("break here")
