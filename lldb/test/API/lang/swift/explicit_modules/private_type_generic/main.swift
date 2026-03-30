import Dylib

struct GenericStruct<T: MyProtocol> {
  let t: T
}

func main() {
  let s = GenericStruct(t: makeInstance())
  print(s) // break here
}

main()
