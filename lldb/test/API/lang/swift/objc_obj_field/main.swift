class C {
    let objcClass = ObjcClass()!
}

func f(_ c: C) {
  print("break here")
}

f(C())

