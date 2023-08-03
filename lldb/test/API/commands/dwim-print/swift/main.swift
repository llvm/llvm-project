class Object: CustomStringConvertible {
  var description: String {
    let address = unsafeBitCast(self, to: Int.self)
    let hexAddress = String(address, radix: 16)
    return "Object@0x\(hexAddress)"
  }
}

class User {
    var id: Int = 314159265358979322
    var name: String = "Gwendolyn"
    var groups: (admin: Bool, staff: Bool) = (false, true)
}

func main() {
    let object = Object()
    let user = User()
    // break here
    print(object, user)
}

main()
