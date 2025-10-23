// REQUIRES: x86

// Test symtab reading
// RUN: %build --compiler=clang-cl --arch=64 --nodefaultlib -o %t.exe -- %s
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 lldb-test symtab %t.exe --find-symbols-by-regex=".*" | FileCheck %s
// RUN: env LLDB_USE_NATIVE_PDB_READER=0 lldb-test symtab %t.exe --find-symbols-by-regex=".*" | FileCheck %s

struct A {
  void something() {}
};

namespace ns {
template <typename T> struct B {
  struct C {
    static int static_fn() { return 1; }
  };

  int b_func() const { return 3; }
};

struct Dyn {
  virtual ~Dyn() = default;
};

int a_function() { return 1; }
} // namespace ns

void *operator new(unsigned long long n) { return nullptr; }
void operator delete(void *p, unsigned long long i) {}

A global_a;
ns::B<long long>::C global_c;
int global_int;

int main(int argc, char **argv) {
  A a;
  a.something();
  ns::B<int>::C::static_fn();
  ns::B<bool>::C::static_fn();
  ns::B<short> b;
  ns::Dyn dyn;
  return ns::a_function() + b.b_func();
}

// CHECK-DAG: Code {{.*}} main
// CHECK-DAG: Code {{.*}} ?b_func@?$B@F@ns@@QEBAHXZ
// CHECK-DAG: Code {{.*}} ?something@A@@QEAAXXZ
// CHECK-DAG: Code {{.*}} ??_GDyn@ns@@UEAAPEAXI@Z
// CHECK-DAG: Code {{.*}} ??2@YAPEAX_K@Z
// CHECK-DAG: Code {{.*}} ??3@YAXPEAX_K@Z
// CHECK-DAG: Code {{.*}} ?static_fn@C@?$B@H@ns@@SAHXZ
// CHECK-DAG: Code {{.*}} ?a_function@ns@@YAHXZ
// CHECK-DAG: Code {{.*}} ?static_fn@C@?$B@_N@ns@@SAHXZ
// CHECK-DAG: Code {{.*}} ??1Dyn@ns@@UEAA@XZ
// CHECK-DAG: Code {{.*}} ??0Dyn@ns@@QEAA@XZ
// CHECK-DAG: Data {{.*}} ?global_int@@3HA
// CHECK-DAG: Data {{.*}} ??_7Dyn@ns@@6B@
// CHECK-DAG: Data {{.*}} ?global_a@@3UA@@A
// CHECK-DAG: Data {{.*}} ?global_c@@3UC@?$B@_J@ns@@A
