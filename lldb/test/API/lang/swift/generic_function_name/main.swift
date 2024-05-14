class C<T>
{
  func f<U> (_ t: T, _ u: U) {
    print("break here")
  }
}

C<Int>().f(1, 2 as Float)
