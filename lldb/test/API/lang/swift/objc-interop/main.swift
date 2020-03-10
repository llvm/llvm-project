import Dispatch

// The dispatch clang module is either imported as Objective-C or as C
// with blocks, and both variants should work in LLDB's expression evaluator.

func main() {
  let label = "lldbtest"
#if os(Darwin)
  // Due to the way CI is set up the apinotes are not yet available.
  let queue = DispatchQueue(label: label)
  print(queue)
#endif
  print(label) //% self.expect("fr var -- label", substrs=['lldbtest'])
               //% self.expect("expr -- label",   substrs=['lldbtest'])
}

main()
