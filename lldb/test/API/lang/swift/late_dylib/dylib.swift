struct FromDylib {
    let msg = "Hello from the Dylib!"
}

@_silgen_name("f") public func f() {
  let x = FromDylib()
  print(x) // line 7
}
