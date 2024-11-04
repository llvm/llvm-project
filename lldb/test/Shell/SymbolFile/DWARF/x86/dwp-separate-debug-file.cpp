// REQUIRES: lld

// RUN: %clang -target x86_64-pc-linux -gsplit-dwarf -gdwarf-5 -c %s -o %t.dwarf5.o
// RUN: ld.lld %t.dwarf5.o -o %t.dwarf5
// RUN: llvm-dwp %t.dwarf5.dwo -o %t.dwarf5.dwp
// RUN: rm %t.dwarf5.dwo
// RUN: llvm-objcopy --only-keep-debug %t.dwarf5 %t.dwarf5.debug
// RUN: llvm-objcopy --strip-all --add-gnu-debuglink=%t.dwarf5.debug %t.dwarf5
// RUN: %lldb %t.dwarf5 -o "target variable a" -b | FileCheck %s

// Run one time with the index cache enabled to populate the index cache. When
// we populate the index cache we have to parse all of the DWARF debug info
// and it is always available.
// RUN: rm -rf %t.lldb-index-cache
// RUN: %lldb \
// RUN:   -O 'settings set symbols.enable-lldb-index-cache true' \
// RUN:   -O 'settings set symbols.lldb-index-cache-path %t.lldb-index-cache' \
// RUN:   -O 'settings set target.preload-symbols false' \
// RUN:   -o "script lldb.target.modules[0].FindTypes('::A').GetTypeAtIndex(0)" \
// RUN:   -o "statistics dump" \
// RUN:   %t.dwarf5 -b | FileCheck %s -check-prefix=CACHE

// Run again after index cache was enabled, which load the index cache. When we
// load the index cache from disk, we don't have any DWARF parsed yet and this
// can cause us to try and access information in the .dwp directly without
// parsing the .debug_info, but this caused crashes when the DWO files didn't
// have a backlink to the skeleton compile unit. This test verifies that we
// don't crash and that we can find types when using .dwp files.
// RUN: %lldb \
// RUN:   -O 'settings set symbols.enable-lldb-index-cache true' \
// RUN:   -O 'settings set symbols.lldb-index-cache-path %t.lldb-index-cache' \
// RUN:   -O 'settings set target.preload-symbols false' \
// RUN:   -o "script lldb.target.modules[0].FindTypes('::A').GetTypeAtIndex(0)" \
// RUN:   -o "statistics dump" \
// RUN:   %t.dwarf5 -b | FileCheck %s -check-prefix=CACHED

// Now test with DWARF4
// RUN: %clang -target x86_64-pc-linux -gsplit-dwarf -gdwarf-4 -c %s -o %t.dwarf4.o
// RUN: ld.lld %t.dwarf4.o -o %t.dwarf4
// RUN: llvm-dwp %t.dwarf4.dwo -o %t.dwarf4.dwp
// RUN: rm %t.dwarf4.dwo
// RUN: llvm-objcopy --only-keep-debug %t.dwarf4 %t.dwarf4.debug
// RUN: llvm-objcopy --strip-all --add-gnu-debuglink=%t.dwarf4.debug %t.dwarf4
// RUN: %lldb %t.dwarf4 -o "target variable a" -b | FileCheck %s

// Run one time with the index cache enabled to populate the index cache. When
// we populate the index cache we have to parse all of the DWARF debug info
// and it is always available.
// RUN: rm -rf %t.lldb-index-cache
// RUN: %lldb \
// RUN:   -O 'settings set symbols.enable-lldb-index-cache true' \
// RUN:   -O 'settings set symbols.lldb-index-cache-path %t.lldb-index-cache' \
// RUN:   -O 'settings set target.preload-symbols false' \
// RUN:   -o "script lldb.target.modules[0].FindTypes('::A').GetTypeAtIndex(0)" \
// RUN:   -o "statistics dump" \
// RUN:   %t.dwarf4 -b | FileCheck %s -check-prefix=CACHE

// Run again after index cache was enabled, which load the index cache. When we
// load the index cache from disk, we don't have any DWARF parsed yet and this
// can cause us to try and access information in the .dwp directly without
// parsing the .debug_info, but this caused crashes when the DWO files didn't
// have a backlink to the skeleton compile unit. This test verifies that we
// don't crash and that we can find types when using .dwp files.
// RUN: %lldb \
// RUN:   -O 'settings set symbols.enable-lldb-index-cache true' \
// RUN:   -O 'settings set symbols.lldb-index-cache-path %t.lldb-index-cache' \
// RUN:   -O 'settings set target.preload-symbols false' \
// RUN:   -o "script lldb.target.modules[0].FindTypes('::A').GetTypeAtIndex(0)" \
// RUN:   -o "statistics dump" \
// RUN:   %t.dwarf4 -b | FileCheck %s -check-prefix=CACHED

// CHECK: (A) a = (x = 47)

// CACHE: script lldb.target.modules[0].FindTypes('::A').GetTypeAtIndex(0)
// CACHE: struct A {
// CACHE-NEXT: int x;
// CACHE-NEXT: }
// CACHE: "totalDebugInfoIndexSavedToCache": 1

// CACHED: script lldb.target.modules[0].FindTypes('::A').GetTypeAtIndex(0)
// CACHED: struct A {
// CACHED-NEXT: int x;
// CACHED-NEXT: }
// CACHED: "totalDebugInfoIndexLoadedFromCache": 1

struct A {
  int x = 47;
};
A a;
int main() {}
