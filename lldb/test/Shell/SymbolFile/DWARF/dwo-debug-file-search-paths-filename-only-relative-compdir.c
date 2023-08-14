/// Check that when LLDB is looking for a relative DWO it uses the debug search
/// paths setting. If it doesn't find it by adding the whole relative path to
/// of DWO it should try adding just the filename (e.g. main.dwo) to each debug
/// search path.
// RUN: rm -rf %t.compdir/
// RUN: mkdir -p %t.compdir/a/b/
// RUN: cp %s %t.compdir/a/b/main.c
// RUN: cd %t.compdir/a/
/// The produced DWO is named /b/main-main.dwo, with a DW_AT_comp_dir of a/.
// RUN: %clang_host -g -gsplit-dwarf -fdebug-prefix-map=%t.compdir=. b/main.c -o b/main
// RUN: cd ../..
/// Move the DWO file away from the expected location.
// RUN: mv %t.compdir/a/b/*.dwo %t.compdir/
/// LLDB won't find the DWO next to the binary or by adding the relative path
/// to any of the search paths. So it should find the DWO file at
/// %t.compdir/main-main.dwo.
// RUN: %lldb --no-lldbinit %t.compdir/a/b/main \
// RUN:   -O "settings append target.debug-file-search-paths %t.compdir" \
// RUN:   -o "b main" -o "run" -o "p num" --batch 2>&1 | FileCheck %s

// CHECK-NOT: warning: {{.*}}main unable to locate separate debug file (dwo, dwp). Debugging will be degraded.
// CHECK: (int) 5

int num = 5;

int main(void) { return 0; }