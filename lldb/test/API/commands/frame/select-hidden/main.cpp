namespace std {
namespace __1 {
static const char *__impl2() { return "Break here"; }
static const char *__impl1() { return __impl2(); }
static const char *__impl() { return __impl1(); }
static const char *non_impl() { return __impl(); }
} // namespace __1
} // namespace std

int main() {
  std::__1::non_impl();
  __builtin_debugtrap();
}
