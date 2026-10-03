// Definitions of two recursive types. Each type reaches itself through a
// function pointer that returns a pointer to the type, which historically made
// LLDB recurse infinitely while completing the type.
//
// These definitions live in their own translation unit (separate from main.cpp)
// so that LLDB has to complete the recursive type from this unit's debug info.

// Variant 1: 't1' reaches itself through a named struct 's1' holding a function
// pointer that returns a pointer to 't1'.
struct t1;
typedef t1 *t1_ptr;
typedef t1_ptr (*get_t1)();
struct s1 {
  get_t1 get_t1_p;
};
struct t1 {
  s1 *s;
};

// Variant 2: 't2' reaches itself through an anonymous struct holding a function
// pointer that returns a pointer to 't2'.
struct t2;
typedef t2 *t2_ptr;
typedef t2_ptr (*get_t2)();
struct t2 {
  struct {
    get_t2 get_t2_p;
  };
};

t1 global_t1;
t2 global_t2;
