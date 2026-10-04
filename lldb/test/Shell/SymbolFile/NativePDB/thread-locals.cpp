// clang-format off
// REQUIRES: lld, x86, target-windows

// Test that we can display tag types.
// RUN: %clang_cl --target=x86_64-windows-msvc -MDd -Od -Z7 -c /Fo%t.obj -- %s
// RUN: lld-link -debug:full -entry:main %t.obj -out:%t.exe -pdb:%t.pdb
// RUN: lldb-test symbols %t.exe | FileCheck %s

// CHECK: CompileUnit{{.*}}, language = "c++", file = '{{.*}}thread-locals.cpp'
// Can't test for the specific address, but we can check that they're all less than 256 (at most two hex digits).
// CHECK-DAG: Variable{{.*}}, name = "tls1", type = {{.*}} (int), scope = thread local, location = DW_OP_const4u 0x{{[0-9a-f][0-9a-f]?}}, DW_OP_form_tls_address, external
// CHECK-DAG: Variable{{.*}}, name = "tls2", type = {{.*}} (int), scope = thread local, location = DW_OP_const4u 0x{{[0-9a-f][0-9a-f]?}}, DW_OP_form_tls_address, external
// CHECK-DAG: Variable{{.*}}, name = "tls3", type = {{.*}} (long long), scope = thread local, location = DW_OP_const4u 0x{{[0-9a-f][0-9a-f]?}}, DW_OP_form_tls_address
// CHECK: CompileUnit{{.*}}

__declspec(thread) int tls1 = 1;
__declspec(thread) int tls2 = 2;
static __declspec(thread) long long tls3 = 3;

int main() {
  return tls3;
}
