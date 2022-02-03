import NoSwiftmoduleHelper

// The struct is resolved using type metadata and the Swift runtime.
struct S { let i = 0 }

func useTypeFromOtherModule(x: S2) {
  // break here
}

enum NoPayload {
  case first
  case second
}

enum WithPayload {
  case empty
  case with(i: Int)
}

func f<T>(_ t: T) {
  let number = 1                     // CHECK-DAG: (Int) number {{=}} 1
  let array = [1, 2, 3]              // CHECK-DAG: ([Int]) array {{=}} 3 values
  let string = "hello"               // CHECK-DAG: (String) string {{=}} "hello"
  let tuple = (0, 1)                 // CHECK-DAG: (Int, Int) tuple {{=}} (0 = 0, 1 = 1)
  let strct = S()                    // CHECK-DAG: strct {{=}} (i = 0)
  let strct2 = S2()                  // CHECK-DAG: strct2 {{=}} {}{{$}}
  let generic = t                    // CHECK-DAG: (Int) generic {{=}} 23
  let generic_tuple = (t, t)         // CHECK-DAG: generic_tuple {{=}} (0 = 23, 1 = 23)
  let word = 0._builtinWordValue     // CHECK-DAG: word {{=}} 0
  let enum1 = NoPayload.second       // CHECK-DAG: enum1 {{=}}
                                     // FIXME: Fails in swift::reflection::NoPayloadEnumTypeInfo::projectEnumValue: .second
  let enum2 = WithPayload.with(i:42) // CHECK-DAG: enum2 {{=}} with
                                     // CHECK-DAG: i {{=}} 42
  print(number)
  useTypeFromOtherModule(x: S2())
}

f(23)
