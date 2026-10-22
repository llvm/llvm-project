/// Check that LLDB does not emit "GetDIE for DIE {{0x[0-9a-f]+}} is outside of its CU"
/// error message when user is searching for a matching symbol from .debug_names
/// and fail to locate the corresponding .dwo file.

/// -gsplit-dwarf is supported only on Linux.
// REQUIRES: system-linux

// RUN: %clang_host -g -gsplit-dwarf -gpubnames -gdwarf-5 %s -o main
/// Remove the DWO file away from the expected location so that LLDB won't find the DWO next to the binary.
// RUN: rm *.dwo
// RUN: %lldb --no-lldbinit main \
// RUN:   -o "b main" --batch 2>&1 | FileCheck %s

// CHECK: warning: {{.*}}main unable to locate separate debug file (dwo, dwp). Debugging will be degraded.
// CHECK-NOT: main GetDIE for DIE {{0x[0-9a-f]+}} is outside of its CU {{0x[0-9a-f]+}}

int num = 5;

int main(void) { return 0; }
