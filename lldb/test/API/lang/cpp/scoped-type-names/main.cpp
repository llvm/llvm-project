// LLDB_TEST_USED is needed as globals of internal-linkage types (function-local
// types, anonymous namespaces, lambdas) would otherwise not be emitted at all.

// --- Global scope ------------------------------------------------------------
struct GlobalStruct {
  int i;
};
LLDB_TEST_USED GlobalStruct g_global_struct;

enum GlobalEnum { GE };
LLDB_TEST_USED GlobalEnum g_global_enum;

typedef GlobalStruct GlobalTypedef;
LLDB_TEST_USED GlobalTypedef g_global_typedef;

LLDB_TEST_USED struct {
  int i;
} g_global_unnamed_struct;

// --- extern "C" (transparent) ------------------------------------------------
extern "C" {
struct ExternCStruct {
  int i;
};
}
LLDB_TEST_USED ExternCStruct g_extern_c_struct;

// --- Namespaces --------------------------------------------------------------
namespace ns {
struct InNs {
  int i;
};
enum class ScopedEnum { A };
typedef int NsTypedef;
using NsAlias = InNs;
struct {
  int i;
} unnamed_in_ns;

namespace inner {
struct InNested {
  int i;
};
} // namespace inner

inline namespace inl {
struct InInline {
  int i;
};
} // namespace inl

namespace {
struct InAnonInNs {
  int i;
};
} // namespace
} // namespace ns
LLDB_TEST_USED ns::InNs g_in_ns;
LLDB_TEST_USED ns::ScopedEnum g_ns_scoped_enum;
LLDB_TEST_USED ns::NsTypedef g_ns_typedef;
LLDB_TEST_USED ns::NsAlias g_ns_alias;
LLDB_TEST_USED decltype(ns::unnamed_in_ns) g_unnamed_in_ns;
LLDB_TEST_USED ns::inner::InNested g_in_nested_ns;
LLDB_TEST_USED ns::InInline g_in_inline_ns;
LLDB_TEST_USED ns::InAnonInNs g_in_anon_in_ns;

namespace {
struct InAnon {
  int i;
};
} // namespace
LLDB_TEST_USED InAnon g_in_anon;

// --- Records -----------------------------------------------------------------
struct Outer {
  struct Inner {
    int i;
  };
  enum InnerEnum { IE };
  typedef int MemberTypedef;
  struct {
    int i;
  } unnamed_member;
  struct Inner2 {
    struct Innermost {
      int i;
    };
  };
};
LLDB_TEST_USED Outer::Inner g_outer_inner;
LLDB_TEST_USED Outer::InnerEnum g_outer_inner_enum;
LLDB_TEST_USED Outer::MemberTypedef g_outer_member_typedef;
LLDB_TEST_USED decltype(Outer::unnamed_member) g_outer_unnamed_member;
LLDB_TEST_USED Outer::Inner2::Innermost g_outer_innermost;

namespace ns {
struct NsOuter {
  struct Inner {
    int i;
  };
};
} // namespace ns
LLDB_TEST_USED ns::NsOuter::Inner g_ns_outer_inner;

template <typename T> struct Tmpl {
  struct Nested {
    T t;
  };
  template <typename U> struct NestedTmpl {
    T t;
    U u;
  };
};
LLDB_TEST_USED Tmpl<int> g_tmpl;
LLDB_TEST_USED Tmpl<int>::Nested g_tmpl_nested;
LLDB_TEST_USED Tmpl<int>::NestedTmpl<char> g_tmpl_nested_tmpl;

namespace ns {
template <typename T> struct NsTmpl {
  struct Nested {
    T t;
  };
};
} // namespace ns
LLDB_TEST_USED ns::NsTmpl<Outer::Inner>::Nested g_ns_tmpl_nested;

// This global is just used to ensure code is emitted for all lexical blocks,
// which ensures Clang has to emit debug info for the involved types.
int sink;

// --- Functions ---------------------------------------------------------------
void global_func() {
  struct Local {
    int i;
  };
  LLDB_TEST_USED static Local g_func_local;
  enum LocalEnum { LE };
  LLDB_TEST_USED static LocalEnum g_func_local_enum;
  typedef int LocalTypedef;
  LLDB_TEST_USED static LocalTypedef g_func_local_typedef;
  LLDB_TEST_USED static struct {
    int i;
  } g_func_unnamed_local;
  struct A_func {
    struct B {
      int i;
    };
    struct {
      int i;
    } u;
  };
  LLDB_TEST_USED static A_func::B g_func_nested_local;
  LLDB_TEST_USED static decltype(A_func::u) g_func_nested_unnamed_local;
  {
    struct InBlock {
      int i;
    };
    LLDB_TEST_USED static InBlock g_block_local;
    struct A_block {
      struct B {
        int i;
      };
    };
    LLDB_TEST_USED static A_block::B g_block_nested_local;
    sink = 1;
  }
}

namespace ns {
void ns_func() {
  struct NsLocal {
    int i;
  };
  LLDB_TEST_USED static NsLocal g_ns_func_local;
}

void nested_local_func(int, char) {
  struct A_int_char {
    struct B {
      int i;
    };
  };
  LLDB_TEST_USED static A_int_char::B g_nested_local_int_char;
}

void nested_local_func(double) {
  struct A_double {
    struct B {
      int i;
    };
  };
  LLDB_TEST_USED static A_double::B g_nested_local_double;
}

void nested_local_variadic(int, ...) {
  struct A_variadic {
    struct B {
      int i;
    };
  };
  LLDB_TEST_USED static A_variadic::B g_nested_local_variadic;
}

namespace {
void nested_local_anon_ns() {
  struct A_anon_ns {
    struct B {
      int i;
    };
  };
  LLDB_TEST_USED static A_anon_ns::B g_nested_local_anon_ns;
}
} // namespace
inline namespace inl {
void nested_local_inline_ns() {
  struct A_inline_ns {
    struct B {
      int i;
    };
  };
  LLDB_TEST_USED static A_inline_ns::B g_nested_local_inline_ns;
}
} // namespace inl
} // namespace ns

extern "C" void nested_local_extern_c() {
  struct A_extern_c {
    struct B {
      int i;
    };
  };
  LLDB_TEST_USED static A_extern_c::B g_nested_local_extern_c;
}

struct WithMethods {
  static void static_method();
  void method(int) const;
  WithMethods();
  operator int();
};
void WithMethods::static_method() {
  struct InStatic {
    int i;
  };
  LLDB_TEST_USED static InStatic g_static_method_local;
}
void WithMethods::method(int) const {
  struct A_method {
    struct B {
      int i;
    };
  };
  LLDB_TEST_USED static A_method::B g_method_nested_local;
}
WithMethods::WithMethods() {
  struct A_ctor {
    struct B {
      int i;
    };
  };
  LLDB_TEST_USED static A_ctor::B g_ctor_nested_local;
  sink = 1;
}
WithMethods::operator int() {
  struct A_conversion {
    struct B {
      int i;
    };
  };
  LLDB_TEST_USED static A_conversion::B g_conversion_nested_local;
  return 0;
}

namespace ns {
template <typename T> struct TmplWithMethod {
  void method(T);
};
template <typename T> void TmplWithMethod<T>::method(T) {
  struct A_tmpl_class_method {
    struct B {
      int i;
    };
  };
  LLDB_TEST_USED static
      typename A_tmpl_class_method::B g_tmpl_class_method_nested_local;
}
template struct TmplWithMethod<Outer::Inner>;
} // namespace ns

template <typename T> void tmpl_func(T) {
  struct TLocal {
    T t;
  };
  LLDB_TEST_USED static TLocal g_tmpl_func_local;
  struct A_tmpl_func {
    struct B {
      T t;
    };
  };
  LLDB_TEST_USED static typename A_tmpl_func::B g_tmpl_func_nested_local;
}
template void tmpl_func<int>(int);

// Two functions with an identically named (and identically sized) local class.
// The nested types must still be named after their own function.
void collide1() {
  struct Same {
    struct B {
      int i;
    };
  };
  LLDB_TEST_USED static Same::B g_collide1;
}
void collide2() {
  struct Same {
    struct B {
      long l;
    };
  };
  LLDB_TEST_USED static Same::B g_collide2;
}

// --- Lambdas -----------------------------------------------------------------
LLDB_TEST_USED auto g_global_lambda = [](int i) { return i; };

namespace ns {
auto ns_lambda = [] { return 1; };
} // namespace ns
LLDB_TEST_USED auto g_ns_lambda = ns::ns_lambda;

void lambda_func() {
  LLDB_TEST_USED static auto g_func_lambda = [] { return 1; };
}

LLDB_TEST_USED int g_run_lambda = [] {
  struct InLambda {
    int i;
  };
  LLDB_TEST_USED static InLambda g_in_lambda;
  struct A_in_lambda {
    struct B {
      int i;
    };
  };
  LLDB_TEST_USED static A_in_lambda::B g_nested_in_lambda;
  return 0;
}();

namespace ns {
LLDB_TEST_USED int g_run_ns_lambda = [](int) {
  struct A_in_ns_lambda {
    struct B {
      int i;
    };
  };
  LLDB_TEST_USED static A_in_ns_lambda::B g_nested_in_ns_lambda;
  return 0;
}(1);
} // namespace ns

int main() {
  global_func();
  ns::ns_func();
  ns::nested_local_func(1, 'a');
  ns::nested_local_func(1.0);
  ns::nested_local_variadic(1);
  ns::nested_local_anon_ns();
  ns::nested_local_inline_ns();
  nested_local_extern_c();
  WithMethods::static_method();
  WithMethods w;
  w.method(1);
  (void)int(w);
  tmpl_func(1);
  lambda_func();
  collide1();
  collide2();
  return 0;
}
