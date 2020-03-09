@_implementationOnly import SomeLibrary

func stop() {}

struct ContainsTwoInts {
  var wrapped: TwoInts
  var other: Int
}

func test(_ value: TwoInts) {
  let container = ContainsTwoInts(wrapped: value, other: 10)
  let simple = 1
  stop() // break here
}

test(TwoInts(2, 3))
