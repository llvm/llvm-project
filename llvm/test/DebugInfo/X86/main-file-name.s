// RUN: llvm-mc -triple x86_64-unknown-linux-gnu -filetype obj -main-file-name foo.S -g -o %t %s
// RUN: llvm-dwarfdump -v -debug-info %t | FileCheck %s

// CHECK: DW_TAG_compile_unit [1]
// CHECK-NOT: DW_TAG_
// CHECK: DW_AT_name [DW_FORM_string]       ("foo.S")
        

# 1 "foo.S"
# 1 "<built-in>" 1
# 1 "foo.S" 2

foo:
  nop
  nop
  nop
        
