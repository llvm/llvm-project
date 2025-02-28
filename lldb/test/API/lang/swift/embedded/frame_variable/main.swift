struct A {
  let field = 4.5
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


struct GenericStructPair<T, U> {
  let t: T
  let u: U
}

class GenericClassPair<T, U> {
  let t: T
  let u: U

  init(t: T, u: U) {
    self.t = t
    self.u = u
  }
}

enum Either<Left, Right> {
  case left(Left)
  case right(Right)
}


struct Outer {
  struct Inner {
    let value = 99
    struct Innerer {
      let innererValue = 101
    }
  }
}

private struct PrivateType {
  let privateField = 100
}


struct OuterGeneric<T> {
  struct SpecializedInner {
    let t: T
  }

  struct GenericInner<U> {
    let t: T
    let u: U
  }
}

func g() {
  struct FunctionType {
    let funcField = 67
  }
  func f() {
    struct InnerFunctionType {
      let innerFuncField = 8479
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
    let fullMultipayloadEnum2 = FullMultipayloadEnum.two(9.5)
    let bigFullMultipayloadEnum1 = BigFullMultipayloadEnum.one(209, 315)
    let bigFullMultipayloadEnum2 = BigFullMultipayloadEnum.two(452.5, 753.5)
    let sup = Sup()
    let sub = Sub()
    let subSub = SubSub()
    let sup2: Sup = SubSub()
    let gsp = GenericStructPair(t: 42, u: 94.5)
    let gsp2 = GenericStructPair(t: Sup(), u: B())
    let gsp3 = GenericStructPair(t: bigFullMultipayloadEnum1, u: smallMultipayloadEnum2)
    let gcp = GenericClassPair(t: 55.5, u: 9348)
    let either = Either<Int, Double>.left(1234)
    let either2 = Either<Sup, _>.right(gsp3)
    // FIXME: remove the instantiation of Outer (rdar://125258124)
    let outer = Outer()
    let inner = Outer.Inner()
    let innerer = Outer.Inner.Innerer()
    let privateType = PrivateType()
    // FIXME: remove the instantiation of OuterGeneric (rdar://125258124)
    let outerGeneric = OuterGeneric<Int>()
    let specializedInner = OuterGeneric<Int>.SpecializedInner(t: 837)
    let genericInner = OuterGeneric<Int>.GenericInner(t: 647, u: 674.5)
    let functionType = FunctionType()
    let innerFunctionType = InnerFunctionType()

    var array: InlineArray<4, Int> = [1, 2, 3, 4]

    // Dummy statement to set breakpoint print can't be used in embedded Swift for now.
    let dummy = A() // break here
    let string = StaticString("Hello") 
    print(string) 
  }
  f()
}

g()
