#include <coroutine>

struct co_sleep {
  co_sleep(int n) : delay{n} {}
  constexpr bool await_ready() const noexcept { return false; }
  void await_suspend(std::coroutine_handle<> h) const noexcept {}
  void await_resume() const noexcept {}
  int delay;
};


struct Task {
  struct promise_type {
    promise_type() = default;
    Task get_return_object() { return {}; }
    std::suspend_never initial_suspend() { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    void unhandled_exception() {}
  };
};

Task foo() noexcept {
  co_await co_sleep{10};
}

int main() {
  foo();
}

// clang++ -S -O0 SampleProfile/Inputs/pseudo-probe-coro-debug-fix.cpp -emit-llvm -Xclang -disable-llvm-passes -std=c++20 -o SampleProfile/pseudo-probe-coro-debug-fix.ll
