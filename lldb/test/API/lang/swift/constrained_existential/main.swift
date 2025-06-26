protocol P1<U> {
  associatedtype U
  func get1() -> U
}

protocol P2<V> {
  associatedtype V
  func get2() -> V
}

struct Impl : P1<Int>, P2<(Int, Int)>  {
  let i = 23
  func get1() -> Int { return i }
  func get2() -> (Int, Int) { return (4, 2) }
}

struct S<T> {
  let s: any P1<T> & P2<(T, T)>
}

func f() {
  let s0: any P1<Int> & P2<(Int, Int)> = Impl()
  let s = S(s: Impl())
  print("break here")
}

f()

