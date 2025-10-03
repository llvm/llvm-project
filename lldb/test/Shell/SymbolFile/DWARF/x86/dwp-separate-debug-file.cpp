// REQUIRES: lld, python

// Now test with DWARF5
// RUN: %clang -target x86_64-pc-linux -gsplit-dwarf -gdwarf-5 -c %s -o %t.dwarf5.o
// RUN: ld.lld %t.dwarf5.o -o %t.dwarf5
// RUN: llvm-dwp %t.dwarf5.dwo -o %t.dwarf5.dwp
// RUN: rm %t.dwarf5.dwo
// RUN: llvm-objcopy --only-keep-debug %t.dwarf5 %t.dwarf5.debug
// RUN: llvm-objcopy --strip-all --add-gnu-debuglink=%t.dwarf5.debug %t.dwarf5
// RUN: %lldb \
// RUN:   -O "log enable dwarf split" \
// RUN:   -o "target variable a" \
// RUN:   -b %t.dwarf5 | FileCheck %s

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

// Make sure that if we load the "%t.dwarf5.debug" file, that we can find and
// load the .dwo file from the .dwp when it is "%t.dwarf5.dwp"
// RUN: %lldb %t.dwarf5.debug -o "b main" -b | FileCheck %s -check-prefix=DEBUG

// Make sure that if we load the "%t.dwarf5" file, that we can find and
// load the .dwo file from the .dwp when it is "%t.dwarf5.debug.dwp"
// RUN: mv %t.dwarf5.dwp %t.dwarf5.debug.dwp
// RUN: %lldb %t.dwarf5 -o "b main" -b | FileCheck %s -check-prefix=DEBUG

// Make sure that if we load the "%t.dwarf5.debug" file, that we can find and
// load the .dwo file from the .dwp when it is "%t.dwarf5.debug.dwp"
// RUN: %lldb %t.dwarf5.debug -o "b main" -b | FileCheck %s -check-prefix=DEBUG

// Make sure that if we remove the .dwp file we see an appropriate error.
// RUN: rm %t.dwarf5.debug.dwp
// RUN: %lldb \
// RUN:   -O "log enable dwarf split" \
// RUN:   -o "b main" \
// RUN:   -b %t.dwarf5 2>&1 | FileCheck %s -check-prefix=NODWP

// RUN: %lldb \
// RUN:   -O "log enable dwarf split" \
// RUN:   -o "b main" \
// RUN:   -b %t.dwarf5.debug 2>&1 | FileCheck %s -check-prefix=NODWP

// Now test with DWARF4
// RUN: %clang -target x86_64-pc-linux -gsplit-dwarf -gdwarf-4 -c %s -o %t.dwarf4.o
// RUN: ld.lld %t.dwarf4.o -o %t.dwarf4
// RUN: llvm-dwp %t.dwarf4.dwo -o %t.dwarf4.dwp
// RUN: rm %t.dwarf4.dwo
// RUN: llvm-objcopy --only-keep-debug %t.dwarf4 %t.dwarf4.debug
// RUN: llvm-objcopy --strip-all --add-gnu-debuglink=%t.dwarf4.debug %t.dwarf4
// RUN: %lldb \
// RUN:   -O "log enable dwarf split" \
// RUN:   -o "target variable a" \
// RUN:   -b %t.dwarf4 | FileCheck %s

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

// Make sure that if we load the "%t.dwarf4.debug" file, that we can find and
// load the .dwo file from the .dwp when it is "%t.dwarf4.dwp"
// RUN: %lldb %t.dwarf4.debug -o "b main" -b | FileCheck %s -check-prefix=DEBUG

// Make sure that if we load the "%t.dwarf4" file, that we can find and
// load the .dwo file from the .dwp when it is "%t.dwarf4.debug.dwp"
// RUN: mv %t.dwarf4.dwp %t.dwarf4.debug.dwp
// RUN: %lldb %t.dwarf4 -o "b main" -b | FileCheck %s -check-prefix=DEBUG

// Make sure that if we load the "%t.dwarf4.debug" file, that we can find and
// load the .dwo file from the .dwp when it is "%t.dwarf4.debug.dwp"
// RUN: %lldb %t.dwarf4.debug -o "b main" -b | FileCheck %s -check-prefix=DEBUG

// Make sure that if we remove the .dwp file we see an appropriate error.
// RUN: rm %t.dwarf4.debug.dwp
// RUN: %lldb \
// RUN:   -O "log enable dwarf split" \
// RUN:   -o "b main" \
// RUN:   -b %t.dwarf4 2>&1 | FileCheck %s -check-prefix=NODWP

// RUN: %lldb \
// RUN:   -O "log enable dwarf split" \
// RUN:   -o "b main" \
// RUN:   -b %t.dwarf4.debug 2>&1 | FileCheck %s -check-prefix=NODWP

// Test if we have a GNU build ID in our main executable and in our debug file,
// and we have a .dwp file that doesn't, that we can still load our .dwp file.
// RUN: %clang -target x86_64-pc-linux -gsplit-dwarf -gdwarf-5 -c %s -o %t.o
// RUN: ld.lld %t.o --build-id=md5 -o %t
// RUN: llvm-dwp %t.dwo -o %t.dwp
// RUN: rm %t.dwo
// RUN: llvm-objcopy --only-keep-debug %t %t.debug
// RUN: llvm-objcopy --strip-all --add-gnu-debuglink=%t.debug %t
// RUN: %lldb \
// RUN:   -O "log enable dwarf split" \
// RUN:   -o "target variable a" \
// RUN:   -b %t | FileCheck %s

// Now move the .debug and .dwp file into another directory so that we can use
// the target.debug-file-search-paths setting to search for the files.
// RUN: mkdir -p %t-debug-info-dir
// RUN: mv %t.dwp %t-debug-info-dir
// RUN: mv %t.debug %t-debug-info-dir
// RUN: %lldb \
// RUN:   -O "log enable dwarf split" \
// RUN:   -O "setting set target.debug-file-search-paths '%t-debug-info-dir'" \
// RUN:   -o "target variable a" \
// RUN:   -b %t | FileCheck %s
// RUN:

// Now move the .debug and .dwp file into another directory so that we can use
// the target.debug-file-search-paths setting to search for the files.
// CHECK: Searching for DWP using:
// CHECK: Found DWP file:
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

// Make sure debug information was loaded by verifying that the
// DEBUG: Breakpoint 1: where = dwp-separate-debug-file.cpp.tmp.dwarf{{[45]}}{{(\.debug)?}}`main + {{[0-9]+}} at dwp-separate-debug-file.cpp:{{[0-9]+}}:{{[0-9]+}}, address = {{0x[0-9a-fA-F]+}}

// Make sure if we load the stripped binary or the debug info file with no .dwp
// nor any .dwo files that we are not able to fine the .dwp or .dwo files.
// NODWP: Searching for DWP using:
// NODWP: Searching for DWP using:
// NODWP: Unable to locate for DWP file for:
// NODWP: unable to locate separate debug file (dwo, dwp). Debugging will be degraded.

struct A {
  int x = 47;
};
A a;
int main() {}
