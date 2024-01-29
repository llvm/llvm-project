import Foo

class MyStruct {
  let m0 : Int  = 0
  let pointer : StructPtr? = nil
  let m2 : Int = 2
  let bridged : BridgedPtr? = nil
  let m4 : Int = 4
  let opaque : OpaqueObj? = nil
  let m6 : Int = 6
  let void : VoidPtr? = nil
  let m8 : Int = 8
  init() {}    
}

func use<T>(_ t: T) {}

func main() {
  let mystruct = MyStruct()
  use(mystruct) // break here
}

main()
