#include <cstddef>

struct CxxVirtualBase {
  int a;
  virtual ~CxxVirtualBase(){};
};
struct CxxVirtualParent : CxxVirtualBase {
  int b;
};

int main(int argc, char ** argv) {
  struct CxxBase {
    int a;
    int b;
  };
  struct CxxParent : CxxBase {
    long long c;
    short d;
  };

  std::nullptr_t std_nullptr_t = nullptr;

  bool found_it = false;
  if (std_nullptr_t) {
    found_it = true;
  } else {
    found_it = (bool) 0;
  }

  enum UEnum { kUZero, kUOne, kUTwo };
  enum class SEnum { kSZero, kSOne };

  UEnum u_enum = kUTwo;
  SEnum s_enum = SEnum::kSOne;

  typedef int td_int_t;
  typedef int* td_int_ptr_t;
  typedef int& td_int_ref_t;
  typedef SEnum td_senum_t;
  td_int_t td_int = 13;
  td_int_ptr_t td_int_ptr = &td_int;
  td_int_ref_t td_int_ref = td_int;
  td_senum_t td_senum = s_enum;

  CxxParent parent;
  parent.a = 1;
  parent.b = 2;
  parent.c = 3;
  parent.d = 4;

  CxxBase* base = &parent;

  int arr[] = {1, 2, 3, 4, 5};
  int* ptr = arr;

  // BREAK(TestCxxStaticCast)
  // BREAK(TestCxxReinterpretCast)

  CxxVirtualParent v_parent; // Set a breakpoint here
  v_parent.a = 1;
  v_parent.b = 2;
  CxxVirtualBase* v_base = &v_parent;

  // BREAK(TestCxxDynamicCast)
  return 0; // Set a breakpoint here
}
