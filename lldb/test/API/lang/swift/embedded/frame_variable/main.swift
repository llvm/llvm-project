struct A {
  let field = 4.2
}

struct B {
  let a = A()
  let b = 123456
}

// TODO: test enums when "rdar://119343683 (Embedded Swift trivial case enum fails to link)" is solved
// // Enum with a single non-payload case.
// enum TrivialEnum {
//   case theCase
// }

// // Enum with 2 or more non-payload cases and no payload cases
// enum NonPayloadEnum {
//   case one
//   case two
// }

// // Enum with 1 payload case and zero or more non-payload cases
// enum SinglePayloadEnum {
//   case nonPayloadOne
//   case payload(B)
//   case nonPayloadTwo
// }

class Sup {
  var supField: Int8 = 42
}

class Sub: Sup {
  var subField = B()
}

class SubSub: Sub {
  var subSubField = A()
}

let varB = B()
let tuple = (A(), B())
// let trivial = TrivialEnum.theCase
// let nonPayload1 = NonPayloadEnum.one
// let nonPayload2 = NonPayloadEnum.two
// let singlePayload = SinglePayloadEnum.payload(B())
// let emptySinglePayload = SinglePayloadEnum.nonPayloadTwo
let sup = Sup()
let sub = Sub()
let subSub = SubSub()
let sup2: Sup = SubSub()

// Dummy statement to set breakpoint print can't be used in embedded Swift for now.
let dummy = A() // break here

