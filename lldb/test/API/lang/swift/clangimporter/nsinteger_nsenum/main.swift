import Foundation
import Enum

func main(_ e1 : ObjCEnum, _ e2 : ObjCEnum) {
  print("break here")
}

let e1 : ObjCEnum = .eCase1
let e2 : ObjCEnum = .eCase2
main(e1, e2)
