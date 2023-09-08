/// Check that LLDB can find a relative DWO file next to a binary just using the
/// filename of that DWO. For example "main.dwo" not "a/b/main.dwo".
// RUN: rm -rf %t.compdir/
// RUN: mkdir -p %t.compdir/a/b/
// RUN: cp %s %t.compdir/a/b/main.c
// RUN: cd %t.compdir/a/
/// The produced DWO is named b/main-main.dwo, with a DW_AT_comp_dir of a/.
// RUN: %clang_host -g -gsplit-dwarf -fdebug-prefix-map=%t.compdir=. b/main.c -o b/main
// RUN: cd ../..
/// Move binary and DWO out of expected locations.
// RUN: mv %t.compdir/a/b/main %t.compdir/
// RUN: mv %t.compdir/a/b/*.dwo %t.compdir/
// RUN: %lldb --no-lldbinit %t.compdir/main \
// RUN:   -o "b main" -o "run" -o "p num" --batch 2>&1 | FileCheck %s

// CHECK-NOT: warning: {{.*}}main unable to locate separate debug file (dwo, dwp). Debugging will be degraded.
// CHECK: (int) 5

int num = 5;

int main(void) { return 0; }