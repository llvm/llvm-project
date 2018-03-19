RUN: dsymutil -y %p/dummy-debug-map.map -oso-prepend-path %p/../Inputs/scattered-reloc/ -f -o - | llvm-dwarfdump -debug-info - | FileCheck %s

// See Inputs/scattered-reloc/scattered.s to see how this test
// actually works.
int bar = 42;

CHECK: DW_TAG_variable
CHECK-NOT: DW_TAG
CHECK: DW_AT_name{{.*}}
"bar" CHECK - NOT : DW_TAG
                        CHECK : DW_AT_location{{.*}}(DW_OP_addr 0x10010)
