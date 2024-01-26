// REQUIRES: lld

// RUN: %clang -target x86_64-pc-linux -gsplit-dwarf -g -c %s -o %t.o
// RUN: ld.lld %t.o -o %t
// RUN: llvm-dwp %t.dwo -o %t.dwp
// RUN: rm %t.dwo
// RUN: llvm-objcopy --only-keep-debug %t %t.debug
// RUN: llvm-objcopy --strip-all --add-gnu-debuglink=%t.debug %t
// RUN: %lldb %t -o "target variable a" -b | FileCheck %s

// Run one time with the index cache enabled to populate the index cache. When
// we populate the index cache we have to parse all of the DWARF debug info
// and it is always available.
// RUN: rm -rf %T/lldb-index-cache
// RUN: %lldb \
// RUN:   -O 'settings set symbols.enable-lldb-index-cache true' \
// RUN:   -O 'settings set symbols.lldb-index-cache-path %T/lldb-index-cache' \
// RUN:   -O 'settings set target.preload-symbols false' \
// RUN:   -o "script lldb.target.modules[0].FindTypes('::A').GetTypeAtIndex(0)" \
// RUN:   -o "statistics dump" \
// RUN:   %t -b | FileCheck %s -check-prefix=CACHE

// Run again after index cache was enabled, which load the index cache. When we
// load the index cache from disk, we don't have any DWARF parsed yet and this
// can cause us to try and access information in the .dwp directly without
// parsing the .debug_info, but this caused crashes when the DWO files didn't
// have a backlink to the skeleton compile unit. This test verifies that we
// don't crash and that we can find types when using .dwp files.
// RUN: %lldb \
// RUN:   -O 'settings set symbols.enable-lldb-index-cache true' \
// RUN:   -O 'settings set symbols.lldb-index-cache-path %T/lldb-index-cache' \
// RUN:   -O 'settings set target.preload-symbols false' \
// RUN:   -o "script lldb.target.modules[0].FindTypes('::A').GetTypeAtIndex(0)" \
// RUN:   -o "statistics dump" \
// RUN:   %t -b | FileCheck %s -check-prefix=CACHED

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
