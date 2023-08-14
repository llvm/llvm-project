/// Check that LLDB uses the paths in target.debug-file-search-paths to find
/// split DWARF files with DW_AT_comp_dir set to some non-relative path, when
/// the program file and DWO have been moved and/or we're executing from another
/// directory. Specifically when the DWO is not at it's "name", here we move
/// it to %t.compdir/main-main.dwo and it's name is c/d/main-main.dwo.
// RUN: rm -rf %t.compdir/ %t.e/
// RUN: mkdir -p %t.compdir/a/b/c/d/
// RUN: cp %s %t.compdir/a/b/c/d/main.c
// RUN: cd %t.compdir/a/b/
/// The produced DWO is named c/d/main-main.dwo, with a non-relative
/// DW_AT_comp_dir of <pathtobuild>/a/b
// RUN: %clang_host -g -gsplit-dwarf c/d/main.c -o c/d/main
// RUN: cd ../../..
/// Move the program.
// RUN: mv %t.compdir/a/b/c/d/main %t.compdir/a/
/// Move the DWO.
// RUN: mv %t.compdir/a/b/c/d/*.dwo %t.compdir
/// Debug it from yet another path.
// RUN: mkdir -p %t.e/
// RUN: cd %t.e
/// LLDB should find in %t.compdir.
// RUN: %lldb --no-lldbinit %t.compdir/a/main \
// RUN:   -O "settings append target.debug-file-search-paths %t.compdir" \
// RUN:   -o "b main" -o "run" -o "p num" --batch 2>&1 | FileCheck %s

// CHECK-NOT: warning: {{.*}}main unable to locate separate debug file (dwo, dwp). Debugging will be degraded.
// CHECK: (int) 5

int num = 5;

int main(void) { return 0; }