struct A<First, Second> {
  class B<Third> {
    func f<T, U>(t: T, u: U) {
      print(1) // break here
    }
  }
}

class D<T> {
  func f<T>(t: T) {
    print(1) // break here
  }
}

func a<T>(t: T) {
  func b<U>(u: U) {
    print(1) // break here
  }
  b(u: 4.2)
}

func c<T>(t: T) {
  func d<T>(t: T) {
    print(1) // break here
  }
  d(t: 42) // break here
}

let b = A<Int, Double>.B<Bool>();
b.f(t: "Hello", u: [1, 2, 3])

let d = D<Int>()
d.f(t: "Hello")

a(t: true)

c(t: "Hello")

