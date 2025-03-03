struct S<T> {
  let t: T
}

enum EGenericS<T> {
  indirect case s(S<T>)
}

class C<T> {
  init(t: T) { self.t = t }
  let t: T
}

struct LargeS<T> {
  let t: T
  let space = (0, 1, 2, 3, 4, 5, 6, 7)
}

enum EGenericLargeS<T> {
  indirect case s(LargeS<T>)
}

enum EGenericC<T> {
  indirect case c(C<T>)
}

enum EGenericMulti<T> {
  indirect case s(S<T>)
  indirect case c(C<T>)
}

enum ETuple {
  indirect case a(Int, String)
  indirect case b(Int)
  case c(Int)
  case d
}

enum ETree<T> {
  indirect case node(ETree<T>, ETree<T>)
  case leaf(T)
}

func main() {
  let generic_s = EGenericS<Int>.s(S(t: 123))
  let generic_large_s = EGenericLargeS<Int>.s(LargeS(t: 123))
  let generic_c = EGenericC<Int>.c(C(t: 123))
  let multi_s = EGenericMulti<Int>.s(S(t: 123))
  let multi_c = EGenericMulti<Int>.c(C(t: 123))
  let tuple_a = ETuple.a(23, "hello")
  let tuple_b = ETuple.b(42)
  let tuple_c = ETuple.c(32)
  let tuple_d = ETuple.d
  let tree = ETree<Int>.node(
      ETree<Int>.node(ETree<Int>.leaf(1),
                      ETree<Int>.leaf(2)),
      ETree<Int>.leaf(3))
  print("break here")
}

main()
