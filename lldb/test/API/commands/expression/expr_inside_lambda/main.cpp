#include <cassert>
#include <cstdio>

namespace {
int global_var = -5;
} // namespace

struct Baz {
  virtual ~Baz() = default;

  virtual int baz_virt() = 0;

  int base_base_var = 12;
};

struct Bar : public Baz {
  virtual ~Bar() = default;

  virtual int baz_virt() override {
    base_var = 10;
    return 1;
  }

  int base_var = 15;
};

struct Foo : public Bar {
  int class_var = 9;
  int shadowed = -137;
  int *class_ptr;

  virtual ~Foo() = default;

  virtual int baz_virt() override {
    shadowed = -1;
    return 2;
  }

  void method() {
    int local_var = 137;
    int shadowed;
    class_ptr = &local_var;
    auto lambda = [&shadowed, this, &local_var,
                   local_var_copy = local_var]() mutable {
      int lambda_local_var = 5;
      shadowed = 5;
      class_var = 109;
      --base_var;
      --base_base_var;
      std::puts("break here");

      auto nested_lambda = [this, &lambda_local_var, local_var] {
        std::puts("break here");
        lambda_local_var = 0;
      };

      nested_lambda();
      --local_var_copy;
      std::puts("break here");

      struct LocalLambdaClass {
        int lambda_class_local = -12345;
        Foo *outer_ptr;

        void inner_method() {
          auto lambda = [this] {
            std::puts("break here");
            lambda_class_local = -2;
            outer_ptr->class_var *= 2;
          };

          lambda();
        }
      };

      LocalLambdaClass l;
      l.outer_ptr = this;
      l.inner_method();
    };
    lambda();
  }

  void non_capturing_method() {
    int local = 5;
    int local2 = 10;

    class_var += [=] {
      std::puts("break here");
      return local + local2;
    }();
  }
};

int main() {
  Foo f;
  f.method();
  f.non_capturing_method();
  return global_var;
}
