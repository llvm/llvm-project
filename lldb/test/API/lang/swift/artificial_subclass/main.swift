import ObjectiveC

class Superclass {
  let a = 42
}

class Subclass: Superclass {
  let b = 97

  override init() {
    super.init()
    let c: AnyClass = objc_allocateClassPair(Subclass.self, "DynamicSubclass", 0)!
    objc_registerClassPair(c);
    object_setClass(self, c)
  }
}

let m = Subclass()
print(m) // break here
