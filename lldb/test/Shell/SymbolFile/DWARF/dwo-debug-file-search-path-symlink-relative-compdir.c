/// Check that LLDB uses the paths in target.debug-file-search-paths to find
/// split DWARF files with a relative DW_AT_comp_dir set, when the program file
/// has been moved and/or we're executing it from another directory. Even when
/// the provided search path is actually a symlink to the real location.
// UNSUPPORTED: system-windows
// RUN: rm -rf %t.compdir/ %t.e/
// RUN: mkdir -p %t.compdir/a/b/c/d/
// RUN: cp %s %t.compdir/a/b/c/d/main.c
// RUN: cd %t.compdir/a/b/
/// The produced DWO is named c/d/main-main.dwo, with a DW_AT_comp_dir of a/b.
// RUN: %clang_host -g -gsplit-dwarf -fdebug-prefix-map=%t.compdir=. c/d/main.c -o c/d/main
// RUN: cd ../../..
/// Move only the program, leaving the DWO file in place.
// RUN: mv %t.compdir/a/b/c/d/main %t.compdir/a/
/// Create a symlink to the compliation dir, to use instead of the real path.
// RUN: ln -s %t.compdir %t.symlink_to_compdir
/// Debug it from yet another path.
// RUN: mkdir -p %t.e/
// RUN: cd %t.e
/// DWO should be found by following using symlink + a/b/ + c/d/main-main.dwo.
// RUN: %lldb --no-lldbinit %t.compdir/a/main \
// RUN:   -O "settings append target.debug-file-search-paths %t.symlink_to_compdir" \
// RUN:   -o "b main" -o "run" -o "p num" --batch 2>&1 | FileCheck %s

// CHECK-NOT: warning: {{.*}}main unable to locate separate debug file (dwo, dwp). Debugging will be degraded.
// CHECK: (int) 5

int num = 5;

int main(void) { return 0; }
