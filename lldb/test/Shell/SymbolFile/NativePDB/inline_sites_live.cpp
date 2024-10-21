// clang-format off
// REQUIRES: system-windows

// RUN: %build -o %t.exe -- %s
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t.exe -s \
// RUN:     %p/Inputs/inline_sites_live.lldbinit 2>&1 | FileCheck %s

void use(int) {}

void __attribute__((always_inline)) bar(int param) {
  use(param); // BP_bar
}

void __attribute__((always_inline)) foo(int param) {
  int local = param+1;
  bar(local);
  use(param);
  use(local); // BP_foo
}

int main(int argc, char** argv) {
  foo(argc);
}

// CHECK:      * thread #1, stop reason = breakpoint 1
// CHECK-NEXT:    frame #0: {{.*}}`main [inlined] bar(param=2)
// CHECK:      (lldb) expression param
// CHECK-NEXT: (int) $0 = 2
// CHECK:      * thread #1, stop reason = breakpoint 2
// CHECK-NEXT:    frame #0: {{.*}}`main [inlined] foo(param=1)
// CHECK:      (lldb) expression param
// CHECK-NEXT: (int) $1 = 1
// CHECK-NEXT: (lldb) expression local
// CHECK-NEXT: (int) $2 = 2
