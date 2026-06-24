
struct wrap_int {
  int inner{};
};
[[gnu::abi_tag("OPERATOR")]]
bool operator==(const wrap_int & /*unused*/, const wrap_int & /*unused*/) {
  return true;
}

[[gnu::abi_tag("FOO")]]
static int foo() {
  return 0;
}

struct [[gnu::abi_tag("STATIC_STRUCT")]] StaticStruct {
  [[gnu::abi_tag("FOO", "FOO2")]]
  static int foo() {
    return 10;
  };
};

struct [[gnu::abi_tag("STRUCT")]] Struct {
  [[gnu::abi_tag("FOO")]]
  int foo() {
    return 10;
  };

  bool operator<(int val) { return false; }

  [[gnu::abi_tag("ops")]]
  unsigned int operator[](int val) {
    return val;
  }

  [[gnu::abi_tag("FOO")]]
  ~Struct() {}
};

namespace ns {
struct [[gnu::abi_tag("NAMESPACE_STRUCT")]] NamespaceStruct {
  [[gnu::abi_tag("FOO")]]
  int foo() {
    return 10;
  }
};

[[gnu::abi_tag("NAMESPACE_FOO")]]
void end_with_foo() {}

[[gnu::abi_tag("NAMESPACE_FOO")]]
void foo() {}
} // namespace ns

template <typename Type>
class [[gnu::abi_tag("TEMPLATE_STRUCT")]] TemplateStruct {

  [[gnu::abi_tag("FOO")]]
  void foo() {
    int something = 32;
  }

public:
  void foo_pub() { this->foo(); }

  template <typename ArgType>
  [[gnu::abi_tag("FOO_TEMPLATE")]]
  void foo(ArgType val) {
    val = 20;
  }

  template <typename Ty>
  [[gnu::abi_tag("OPERATOR")]]
  bool operator<(Ty val) {
    return false;
  }

  template <typename Ty>
  [[gnu::abi_tag("operator")]]
  bool operator<<(Ty val) {
    return false;
  }
};

int main() {
  // standalone
  const int res1 = foo();

  // static
  const int res2 = StaticStruct::foo();

  // normal
  {
    Struct normal;
    const int res3 = normal.foo();
    const bool operator_lessthan_res = normal < 10;
  }

  // namespace
  ns::NamespaceStruct ns_struct;
  const int res4 = ns_struct.foo();

  ns::foo();

  // template struct
  TemplateStruct<int> t_struct;
  t_struct.foo_pub();
  const long into_param = 0;
  t_struct.foo(into_param);

  const bool t_ops_lessthan = t_struct < 20;
  const bool t_ops_leftshift = t_struct << 30;

  // standalone operator
  wrap_int lhs;
  wrap_int rhs;
  const bool res6 = lhs == rhs;
  return 0;
}
