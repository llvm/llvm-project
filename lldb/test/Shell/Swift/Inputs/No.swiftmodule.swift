import NoSwiftmoduleHelper

func useTypeFromOtherModule(x: S2) {
  // break here
}


func f<T>(_ t: T) {
  let strct2 = S2()                  // CHECK-DAG: strct2 {{=}} {}{{$}}
  print(strct2)
  useTypeFromOtherModule(x: S2())
}

f(23)
