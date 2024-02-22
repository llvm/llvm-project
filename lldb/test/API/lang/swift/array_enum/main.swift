enum x : String {
  case patatino
}

struct y {
  var z: x?
}

func main() -> Int {
  var a = y()
  a.z = x.patatino
  var j = [a]
  return 0 //%self.expect('frame var -d run-target a', substrs=['(a.y) a = (z = patatino)'])
           //%self.expect('expr -d run-target -- a', substrs=['(a.y) $R0 = (z = patatino)'])
           //%self.expect('frame var -d run-target j', substrs=['[0] = (z = patatino)'])
           //%self.expect('expr -d run-target -- j', substrs=['[0] = (z = patatino)'])
}

let _ = main()
