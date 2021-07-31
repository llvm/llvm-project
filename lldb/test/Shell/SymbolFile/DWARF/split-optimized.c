// Test that optimized flag is properly included in DWARF.

// ObjectFileELF::ApplyRelocations does not implement arm32.
// XFAIL: target-arm && linux-gnu

// RUN: %clang_host %s -fno-standalone-debug -glldb \
// RUN:   -gdwarf-5 -gpubnames -gsplit-dwarf -O3 -c -o %t1.o

// RUN: llvm-dwarfdump %t1.o | FileCheck %s --check-prefix DWARFDUMP_O
// RUN: llvm-dwarfdump %t1.dwo | FileCheck %s --check-prefix DWARFDUMP_DWO
// RUN: %lldb -b -o 'script lldb.SBDebugger.Create().CreateTarget("%t1.o").FindFunctions("main",lldb.eFunctionNameTypeAuto).GetContextAtIndex(0).GetFunction().GetIsOptimized()' | FileCheck %s

// DWARFDUMP_O-NOT: DW_AT_APPLE_optimized
//
// DWARFDUMP_DWO: DW_TAG_compile_unit
// DWARFDUMP_DWO-NOT: DW_TAG_
// DWARFDUMP_DWO: DW_AT_APPLE_optimized	(true)

// CHECK: (lldb) script lldb.SBDebugger.Create()
// CHECK-NEXT: True

int main(void) { return 0; }
