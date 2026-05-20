@objc public enum TheObjCEnum: UInt32 {
  case foo
  case bar
}

@c public enum TheCEnum: Int32 {
  case foo
  case bar
}

struct Wrap {
  let objcValue: TheObjCEnum
  let cValue: TheCEnum
}

func entry() {
  let v = Wrap(objcValue: .foo, cValue: .foo)
  print("break here")
  _ = v
}

entry()
