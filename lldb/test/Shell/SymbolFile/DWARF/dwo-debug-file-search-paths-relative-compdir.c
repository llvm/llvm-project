/// Check that LLDB uses the paths in target.debug-file-search-paths to find
/// split DWARF files with a relative DW_AT_comp_dir set, when the program file
/// has been moved and/or we're executing it from another directory.
// UNSUPPORTED: system-darwin
// RUN: rm -rf %t.compdir/ %t.e/
// RUN: mkdir -p %t.compdir/a/b/c/d/
// RUN: cp %s %t.compdir/a/b/c/d/main.c
// RUN: cd %t.compdir/a/b/
/// The produced DWO is named c/d/main-main.dwo, with a DW_AT_comp_dir of a/b.
// RUN: %clang_host -g -gsplit-dwarf -fdebug-prefix-map=%t.compdir=. c/d/main.c -o c/d/main
// RUN: cd ../../..
/// Move only the program, leaving the DWO file in place.
// RUN: mv %t.compdir/a/b/c/d/main %t.compdir/a/
/// Debug it from yet another path.
// RUN: mkdir -p %t.e/
// RUN: cd %t.e
/// LLDB won't find the DWO next to the binary or in the current dir, so it
/// should find the DWO file by doing %t.compdir/ + a/b/ + c/d/main-main.dwo.
// RUN: %lldb --no-lldbinit %t.compdir/a/main \
// RUN:   -O "settings append target.debug-file-search-paths %t.compdir" \
// RUN:   -o "b main" -o "run" -o "p num" --batch 2>&1 | FileCheck %s

// CHECK-NOT: warning: {{.*}}main unable to locate separate debug file (dwo, dwp). Debugging will be degraded.
// CHECK: (int) 5

int num = 5;

int main(void) { return 0; }
