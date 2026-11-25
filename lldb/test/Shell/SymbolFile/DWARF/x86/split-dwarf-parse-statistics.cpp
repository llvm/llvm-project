// Test that the statistics include the time spent parsing a split DWARF unit,
// and that resetting the statistics clears it.

// REQUIRES: lld, python

// With .debug_names, LLDB parses the .dwo unit only when a lookup needs it.
// Dumping the line table and the statistics once before the first reset does
// the skeleton-side parsing, so only the .dwo unit is parsed after the reset.

// RUN: %clangxx %s -c -o %t.o --target=x86_64-pc-linux -gdwarf-5 -gsplit-dwarf -gpubnames
// RUN: ld.lld %t.o -o %t
// RUN: rm -f %t.dwp
// RUN: llvm-dwarfdump --debug-names %t | FileCheck %s --check-prefix=NAMES
// RUN: %lldb -b -o "settings set target.preload-symbols false" \
// RUN:   -o "settings set symbols.enable-lldb-index-cache false" \
// RUN:   -o "target create %t" -o "target modules dump line-table %s" \
// RUN:   -o "statistics dump" -o "script lldb.target.ResetStatistics()" \
// RUN:   -o "statistics dump" -o "target variable foo" -o "statistics dump" \
// RUN:   -o "script lldb.target.ResetStatistics()" -o "statistics dump" \
// RUN:   | FileCheck %s

// RUN: llvm-dwp %t.dwo -o %t.dwp
// RUN: rm %t.dwo
// RUN: %lldb -b -o "settings set target.preload-symbols false" \
// RUN:   -o "settings set symbols.enable-lldb-index-cache false" \
// RUN:   -o "target create %t" -o "target modules dump line-table %s" \
// RUN:   -o "statistics dump" -o "script lldb.target.ResetStatistics()" \
// RUN:   -o "statistics dump" -o "target variable foo" -o "statistics dump" \
// RUN:   -o "script lldb.target.ResetStatistics()" -o "statistics dump" \
// RUN:   | FileCheck %s

// NAMES: CU count: 1
// NAMES: String: {{.*}} "foo"
// NAMES: Tag: DW_TAG_variable

// CHECK: (lldb) script lldb.target.ResetStatistics()
// CHECK: (lldb) statistics dump
// CHECK: "debugInfoParseTime": 0,
// CHECK: "loadedDwoFileCount": 0,
// CHECK: "totalDebugInfoParseTime": 0,
// CHECK: (lldb) target variable foo
// CHECK-NEXT: (int) foo = 0
// CHECK: (lldb) statistics dump
// CHECK: "debugInfoParseTime": {{[1-9]|0\.[0-9]*[1-9]}}
// CHECK: "loadedDwoFileCount": 1,
// CHECK: "totalDebugInfoParseTime": {{[1-9]|0\.[0-9]*[1-9]}}
// CHECK: (lldb) script lldb.target.ResetStatistics()
// CHECK: (lldb) statistics dump
// CHECK: "debugInfoParseTime": 0,
// CHECK: "loadedDwoFileCount": 1,
// CHECK: "totalDebugInfoParseTime": 0,

// Enough DIEs that parsing the unit takes well over the 1 us resolution of
// the statistics.
template <int N> struct Chain : Chain<N - 1> {};
template <> struct Chain<0> {};
Chain<100> chain;

int foo;
extern "C" void _start() {}
