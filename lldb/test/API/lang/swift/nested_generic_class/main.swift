public class A<T> {
  public class B<U> {
    func f() {
        print(1) // break here
    }
  }
}

class C {
  class D<T> {
    func f() {
      print(1) // break here
    }
  }
}

class F {
  class G<T> {
    class H {
      func f() {
        print(1) // break here
      }
    }
  }
}

class I {
  class J<T, U> {
    func f() {
      print(1) // break here
    }
  }
}

class K {
  class L {
    class M<T, U> {
      func f() {
        print(1) // break here
      }
    }
  }
}

A<Int>.B<String>().f()
C.D<Double>().f()
F.G<Bool>.H().f()
I.J<String, Int>().f()
K.L.M<Double, Bool>().f()
