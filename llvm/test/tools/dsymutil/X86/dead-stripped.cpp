// RUN: llvm-dsymutil -f -y %p/dummy-debug-map.map -oso-prepend-path %p/../Inputs/dead-stripped -o - | llvm-dwarfdump - --debug-info | FileCheck %s

// The test was compiled with:
// clang++ -O2 -g -c dead-strip.cpp -o 1.o

// The goal of the test is to exercise dsymutil's behavior in presence of
// functions/variables that have been dead-stripped by the linker but
// that are still present in the linked debug info (in this case because
// they have been DW_TAG_import'd in another namespace).

// Everything in the N namespace bellow doesn't have a debug map entry, and
// thus is considered dead (::foo() has a debug map entry, otherwse dsymutil
// would just drop the CU altogether).

namespace N {
int blah = 42;
// This is actually a dsymutil-classic bug that we reproduced
// CHECK: DW_TAG_variable
// CHECK-NOT: DW_TAG
// CHECK: DW_AT_location

__attribute__((always_inline)) int foo() { return blah; }
// CHECK: DW_TAG_subprogram
// CHECK-NOT: DW_AT_low_pc
// CHECK-NOT: DW_AT_high_pc
// CHECK: DW_AT_frame_base

// CHECK: DW_TAG_subprogram

int bar(unsigned i) {
	int val = foo();
	if (i)
		return val + bar(i-1);
	return foo();
}
// CHECK: DW_TAG_subprogram
// CHECK-NOT: DW_AT_low_pc
// CHECK-NOT: DW_AT_high_pc
// CHECK: DW_AT_frame_base
// CHECK-NOT: DW_AT_location
// CHECK-NOT: DW_AT_low_pc
// CHECK-NOT: DW_AT_high_pc

}
// CHECK: TAG_imported_module
using namespace N;

void foo() {}
