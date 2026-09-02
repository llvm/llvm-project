// RUN: %libomptarget-compile-generic -fopenmp-version=51
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic

extern "C" int printf(const char *, ...);
template <typename T> class A {
protected:
  T X;
  T Y;

public:
  A(T x, T y) : X{x}, Y{y} {};
};

template <typename T> class B : public A<T> {
  using A<T>::X;
  using A<T>::Y;

public:
  T res;

  B(T x, T y) : A<T>(x, y), res{0} {};

  void run(void) {
#pragma omp target map(res)
    { res = X + Y; }
  }
};

class X {
protected:
  int A;

public:
  X(int a) : A{a} {};
};
class Y : public X {
  using X::A;

protected:
  int B;

public:
  Y(int a, int b) : X(a), B{b} {};
};
class Z : public Y {
  using X::A;
  using Y::B;

public:
  int res;
  Z(int a, int b) : Y(a, b), res{0} {};
  void run(void) {
#pragma omp target map(res)
    { res = A + B; }
  }
};
struct descriptor {
  int A;
  int C;
};

class BASE {};

class C : public BASE {
public:
  void bar(descriptor &d) {
    auto Asize = 4;
    auto Csize = 4;

#pragma omp target data map(from : d.C)
    {
#pragma omp target teams firstprivate(Csize)
      d.C = 1;
    }
#pragma omp target map(from : d.A)
    d.A = 3;
  }
};

int main(int argc, char *argv[]) {
  B<int> b(2, 3);
  b.run();
  // CHECK: 5
  printf("b.res = %d \n", b.res);
  Z c(2, 3);
  c.run();
  // CHECK: 5
  printf("c.res = %d \n", c.res);

  descriptor d;
  C z;
  z.bar(d);
  // CHECK 1
  printf("%d\n", d.C);
  // CHECK 3
  printf("%d\n", d.A);
}
