struct DefaultMirror {
  var a = 12
  var b = 24
}

struct CustomMirror : CustomReflectable {
  var a = 12
  var b = 24
  
  public var customMirror: Mirror {
    get { return Mirror(self, children: ["c" : a + b]) }
  }
}

struct CustomSummary : CustomStringConvertible, CustomDebugStringConvertible {
  var a = 12
  var b = 24
  
  var description: String {
    return "CustomStringConvertible"
  }
  var debugDescription: String {
    return "CustomDebugStringConvertible" // Breakpoint in debugDescription
  }
}

func main() {
  var dm = DefaultMirror()
  var cm = CustomMirror()
  var cs = CustomSummary()
  var patatino = "foo"
 
  print("yay I am done!") // Break here to run tests.
}

main()
