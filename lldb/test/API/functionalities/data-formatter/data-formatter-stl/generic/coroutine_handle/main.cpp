#if defined(USE_LIBSTDCPP)
#include <bits/c++config.h>
// glibc++ >= 11 and c++20
#if defined(_GLIBCXX_RELEASE) && _GLIBCXX_RELEASE >= 11
#include <coroutine>
#define HAS_CPP_COROUTINES 1
#endif
#endif

// libc++ always has 'coroutine' feature.
#if defined(USE_LIBCPP)
#include <coroutine>
#define HAS_CPP_COROUTINES 1
#endif

bool is_implementation_supported() {
#ifdef HAS_CPP_COROUTINES
  return true;
#else
  return false;
#endif
}

#ifdef HAS_CPP_COROUTINES
// `int_generator` is a stripped down, minimal coroutine generator
// type.
struct int_generator {
  struct promise_type {
    int current_value = -1;

    auto get_return_object() {
      return std::coroutine_handle<promise_type>::from_promise(*this);
    }
    auto initial_suspend() { return std::suspend_always(); }
    auto final_suspend() noexcept { return std::suspend_always(); }
    auto return_void() { return std::suspend_always(); }
    void unhandled_exception() { __builtin_unreachable(); }
    auto yield_value(int v) {
      current_value = v;
      return std::suspend_always();
    }
  };

  std::coroutine_handle<promise_type> hdl;

  int_generator(std::coroutine_handle<promise_type> h) : hdl(h) {}
  ~int_generator() { hdl.destroy(); }
};

int_generator my_generator_func() { co_yield 42; }

// This is an empty function which we call just so the debugger has
// a place to reliably set a breakpoint on.
void empty_function_so_we_can_set_a_breakpoint() {}

#endif // HAS_CPP_COROUTINES

int main() {
  bool is_supported = is_implementation_supported();
#ifdef HAS_CPP_COROUTINES
  int_generator gen = my_generator_func();
  std::coroutine_handle<> type_erased_hdl = gen.hdl;
  std::coroutine_handle<int> incorrectly_typed_hdl =
      std::coroutine_handle<int>::from_address(gen.hdl.address());
  gen.hdl.resume();                            // Break at initial_suspend
  gen.hdl.resume();                            // Break after co_yield
  empty_function_so_we_can_set_a_breakpoint(); // Break at final_suspend
  return 0;
#else
  return 0; // Break at initial_suspend
#endif // HAS_CPP_COROUTINES
}
