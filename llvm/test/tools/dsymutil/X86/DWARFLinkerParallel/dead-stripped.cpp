// RUN: dsymutil --linker llvm --no-odr -f -y %p/../dummy-debug-map.map -oso-prepend-path \
// RUN: %p/../../Inputs/dead-stripped -o - | llvm-dwarfdump - --debug-info | \
// RUN: FileCheck %s --implicit-check-not \
// RUN: "{{DW_AT_low_pc|DW_AT_high_pc|DW_AT_location|DW_TAG|NULL}}"
//
// RUN: dsymutil --linker llvm -f -y %p/../dummy-debug-map.map -oso-prepend-path \
// RUN: %p/../../Inputs/dead-stripped -o - | llvm-dwarfdump - --debug-info | \
// RUN: FileCheck %s --implicit-check-not \
// RUN: "{{DW_AT_low_pc|DW_AT_high_pc|DW_AT_location|DW_TAG|NULL}}" \
// RUN: --check-prefixes=CHECK-LLVM

// The test was compiled with:
// clang++ -O2 -g -c dead-strip.cpp -o 1.o

// The goal of the test is to exercise dsymutil's behavior in presence of
// functions/variables that have been dead-stripped by the linker but
// that are still present in the linked debug info (in this case because
// they have been DW_TAG_import'd in another namespace).

// CHECK-LLVM: DW_TAG_compile_unit
// CHECK-LLVM: DW_AT_name{{.*}}"__artificial_type_unit"
// CHECK-LLVM: DW_TAG_namespace
// CHECK-LLVM: DW_AT_name{{.*}}"N"
// CHECK-LLVM: NULL
//
// CHECK-LLVM: DW_TAG_compile_unit
// CHECK-LLVM:   DW_AT_low_pc
// CHECK-LLVM:   DW_AT_high_pc
// CHECK-LLVM: DW_TAG_base_type
// CHECK-LLVM: DW_TAG_imported_module
// CHECK-LLVM: DW_TAG_subprogram
// CHECK-LLVM:   DW_AT_low_pc
// CHECK-LLVM:   DW_AT_high_pc
// CHECK-LLVM: DW_TAG_base_type
// CHECK-LLVM: NULL

// Everything in the N namespace below doesn't have a debug map entry, and
// thus is considered dead (::foo() has a debug map entry, otherwise dsymutil
// would just drop the CU altogether).

// CHECK: DW_TAG_compile_unit
// CHECK:   DW_AT_low_pc
// CHECK:   DW_AT_high_pc
// CHECK:   DW_TAG_namespace
namespace N {
int blah = 42;

__attribute__((always_inline)) int foo() { return blah; }

int bar(unsigned i) {
	int val = foo();
	if (i)
		return val + bar(i-1);
	return foo();
}
}
// CHECK: DW_TAG_base_type
// CHECK: DW_TAG_imported_module
// CHECK: DW_TAG_subprogram
// CHECK:   DW_AT_low_pc
// CHECK:   DW_AT_high_pc
// CHECK: DW_TAG_base_type
// CHECK: NULL

using namespace N;

void foo() {}
