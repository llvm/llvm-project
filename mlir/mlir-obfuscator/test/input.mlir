// String test
func.func @hello() attributes { msg = "HELLO WORLD" } {
  return
}

// Symbol test
func.func @main() {
  %0 = func.call @hello() : () -> ()
  return
}
