import NoSwiftmoduleHelper

// The struct could not possibly be resolved with just the mangled type name.
struct s { let i = 0 }

func useTypeFromOtherModule(x: S2) {
  // break here
}

func f<T>(_ t: T) {
  let number = 1                   // CHECK-DAG: (Int) number = 1
  let array = [1, 2, 3]            // CHECK-DAG: ([Int]) array = 3 values
  let string = "hello"             // CHECK-DAG: (String) string = "hello"
  let tuple = (0, 1)               // CHECK-DAG: (Int, Int) tuple = (0 = 0, 1 = 1)
  let strct = s()                  // CHECK-DAG: strct = {}{{$}}
  let strct2 = S2()                // CHECK-DAG: strct2 = <extracting data from value failed>
  let generic = t                  // CHECK-DAG: (Int) generic = 23
  let generic_tuple = (t, t)       // CHECK-DAG: generic_tuple = (0 = 23, 1 = 23)
  print(number)
  useTypeFromOtherModule(x: S2())
}

f(23)
