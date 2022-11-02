public class A<T> {
  public class B<U> {
  }
}

let foo = A<Int>.B<String>()
print(1) // break here
