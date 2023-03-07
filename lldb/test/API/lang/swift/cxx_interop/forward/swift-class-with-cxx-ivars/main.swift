import ReturnsClass

class SwiftClass {
  let cxxClass = CxxClass()
  let cxxSubclass = InheritedCxxClass()
}

struct SwiftStruct {
  let cxxClass = CxxClass()
  let cxxSubclass = InheritedCxxClass()
}

enum SwiftEnum {
  case first(CxxClass)
  case second(InheritedCxxClass)

}

func main() {
  let swiftClass = SwiftClass()
  let swiftStruct = SwiftStruct()
  let swiftEnum1 = SwiftEnum.first(CxxClass())
  let swiftEnum2 = SwiftEnum.second(InheritedCxxClass())
  print(1) // Set breakpoint here
}
main()
