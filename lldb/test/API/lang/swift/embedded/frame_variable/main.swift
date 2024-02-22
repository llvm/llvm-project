struct A {
  let field = 4.2
}

struct B {
  let a = A()
  let b = 123456
}

// Enum with a single non-payload case.
enum TrivialEnum {
  case theCase
}

// Enum with 2 or more non-payload cases and no payload cases
enum NonPayloadEnum {
  case one
  case two
}

// Enum with 1 payload case and zero or more non-payload cases
enum SinglePayloadEnum {
  case nonPayloadOne
  case payload(B)
  case nonPayloadTwo
}

// A MultiPayloadEnum whose payload has less than 64 bits
enum SmallMultipayloadEnum {
  case empty
  case one(NonPayloadEnum)
  case two(NonPayloadEnum)
}

// A MultiPayloadEnum whose payload has more than 64 bits
enum BigMultipayloadEnum {
  case one(Sup, Sup, Sup)
  case two(B)
}

// A MultiPayloadEnum with no spare bits
enum FullMultipayloadEnum {
  case one(Int)
  case two(Double)
}

// A MultiPayloadEnum whose payload has more than 64 bits and no spare bits
enum BigFullMultipayloadEnum {
  case one(Int, Int)
  case two(Double, Double)
}

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
let trivial = TrivialEnum.theCase
let nonPayload1 = NonPayloadEnum.one
let nonPayload2 = NonPayloadEnum.two
let singlePayload = SinglePayloadEnum.payload(B())
let emptySinglePayload = SinglePayloadEnum.nonPayloadTwo
let smallMultipayloadEnum1 = SmallMultipayloadEnum.one(.two)
let smallMultipayloadEnum2 = SmallMultipayloadEnum.two(.one)
let e1 = Sup()
let e2 = Sup()
e2.supField = 43
let e3 = Sup()
e3.supField = 44
let bigMultipayloadEnum1 = BigMultipayloadEnum.one(e1, e2, e3)
let fullMultipayloadEnum1 = FullMultipayloadEnum.one(120)
let fullMultipayloadEnum2 = FullMultipayloadEnum.two(9.21)
let bigFullMultipayloadEnum1 = BigFullMultipayloadEnum.one(209, 315)
let bigFullMultipayloadEnum2 = BigFullMultipayloadEnum.two(452.2, 753.9)
let sup = Sup()
let sub = Sub()
let subSub = SubSub()
let sup2: Sup = SubSub()

// Dummy statement to set breakpoint print can't be used in embedded Swift for now.
let dummy = A() // break here
let string = StaticString("Hello") 
print(string) 

