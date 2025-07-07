// REQUIRES: lld

// Test if we build a mixed binary where one .o file has a .debug_names and
// another doesn't have one, that we save a full or partial index cache.
// Previous versions of LLDB would have ManualDWARFIndex.cpp that would save out
// an index cache to the same file regardless of wether the index cache was a
// full DWARF manual index, or just the CUs and TUs that were missing from any
// .debug_names tables. If the user had a .debug_names table and debugged once
// with index caching enabled, then debugged again but set the setting to ignore
// .debug_names ('settings set plugin.symbol-file.dwarf.ignore-file-indexes 1')
// this could cause LLDB to load the index cache from the previous run which
// was incomplete and it only contained the manually indexed DWARF from the run
// where we used .debug_names, but it would now load it as if it were the
// complete DWARF index.

// Test that if we don't have .debug_names, that we save a full DWARF index.
// RUN: %clang -target x86_64-pc-linux -gsplit-dwarf -gdwarf-5 -DMAIN=1 -c %s -o %t.main.o
// RUN: %clang -target x86_64-pc-linux -gsplit-dwarf -gdwarf-5 -DMAIN=0 -c %s -o %t.foo.o
// RUN: ld.lld %t.main.o %t.foo.o -o %t.nonames
// RUN: llvm-dwp %t.main.dwo %t.foo.dwo -o %t.nonames.dwp
// RUN: rm %t.main.dwo %t.foo.dwo
// Run one time with the index cache enabled to populate the index cache. When
// we populate the index cache we have to parse all of the DWARF debug info
// and it is always available.
// RUN: rm -rf %t.lldb-index-cache
// RUN: %lldb \
// RUN:   -O 'settings set symbols.enable-lldb-index-cache true' \
// RUN:   -O 'settings set symbols.lldb-index-cache-path %t.lldb-index-cache' \
// RUN:   -O 'settings set target.preload-symbols true' \
// RUN:   %t.nonames -b

// Make sure there is a file with "dwarf-index-full" in its filename
// RUN: ls %t.lldb-index-cache | FileCheck %s -check-prefix=FULL
// FULL: {{dwp-index-cache.cpp.tmp.nonames.*-dwarf-index-full-}}

// Test that if we have one .o file with .debug_names and one without, that we
// save a partial DWARF index.
// RUN: %clang -target x86_64-pc-linux -gsplit-dwarf -gdwarf-5 -DMAIN=1 -c %s -o %t.main.o -gpubnames
// RUN: %clang -target x86_64-pc-linux -gsplit-dwarf -gdwarf-5 -DMAIN=0 -c %s -o %t.foo.o
// RUN: ld.lld %t.main.o %t.foo.o -o %t.somenames
// RUN: llvm-dwp %t.main.dwo %t.foo.dwo -o %t.somenames.dwp
// RUN: rm %t.main.dwo %t.foo.dwo
// Run one time with the index cache enabled to populate the index cache. When
// we populate the index cache we have to parse all of the DWARF debug info
// and it is always available.
// RUN: rm -rf %t.lldb-index-cache
// RUN: %lldb \
// RUN:   -O 'settings set symbols.enable-lldb-index-cache true' \
// RUN:   -O 'settings set symbols.lldb-index-cache-path %t.lldb-index-cache' \
// RUN:   -O 'settings set target.preload-symbols true' \
// RUN:   %t.somenames -b

// Make sure there is a file with "dwarf-index-full" in its filename
// RUN: ls %t.lldb-index-cache | FileCheck %s -check-prefix=PARTIAL
// PARTIAL: {{dwp-index-cache.cpp.tmp.somenames.*-dwarf-index-partial-}}

#if MAIN
extern int foo();
int main() { return foo(); }
#else
int foo() { return 0; }
#endif
