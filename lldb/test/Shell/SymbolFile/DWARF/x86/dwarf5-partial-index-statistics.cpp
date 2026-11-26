// Test that the statistics include the time spent indexing the units that
// .debug_names does not cover, and that resetting the statistics clears it.

// REQUIRES: lld, python

// RUN: %clangxx %s -c -o %t-1.o --target=x86_64-pc-linux -DONE -gdwarf-5 -gpubnames
// RUN: %clangxx %s -c -o %t-2.o --target=x86_64-pc-linux -DTWO -gdwarf-5 -gno-pubnames
// RUN: ld.lld %t-1.o %t-2.o -o %t
// RUN: llvm-dwarfdump --debug-info --debug-names %t | FileCheck %s --check-prefix DWARF
// RUN: %lldb -b -o "settings set target.preload-symbols false" \
// RUN:   -o "settings set symbols.enable-lldb-index-cache false" \
// RUN:   -o "target create %t" -o "target variable two::foo" \
// RUN:   -o "statistics dump" -o "script lldb.target.ResetStatistics()" \
// RUN:   -o "statistics dump" | FileCheck %s

// DWARF-COUNT-2: Compile Unit:
// DWARF: CU count: 1

// CHECK: (int) two::foo = 0
// CHECK: "debugInfoIndexTime": {{[1-9]|0\.[0-9]*[1-9]}}
// CHECK: "totalDebugInfoIndexTime": {{[1-9]|0\.[0-9]*[1-9]}}
// CHECK: "debugInfoIndexTime": 0,
// CHECK: "totalDebugInfoIndexTime": 0,

#ifdef ONE
namespace one {
int foo;
} // namespace one
extern "C" void _start() {}
#else
namespace two {
int foo;
} // namespace two
#endif
