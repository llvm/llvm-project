struct Base {
  int m_base_val = 42;
};

LLDB_DYLIB_EXPORT struct Foo : public Base {
  int m_derived_val = 137;
} global_foo;
