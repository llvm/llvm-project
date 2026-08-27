// The recursive types are only forward-declared here; their full (recursive)
// definition lives in recursive_types.cpp. This forces LLDB to complete the
// types from the other translation unit when evaluating the test expressions.
struct t1;
typedef t1 *t1_ptr;
struct t2;
typedef t2 *t2_ptr;

extern t1 global_t1;
extern t2 global_t2;

int main() {
  t1_ptr p1 = &global_t1;
  t2_ptr p2 = &global_t2;
  return 0; // break here
}
