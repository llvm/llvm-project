# RUN: llvm-mc -filetype=obj -triple=aarch64-none-linux-gnu %s | llvm-readobj -r - | FileCheck %s
  
# CHECK: Relocations [
# CHECK:   Section {{.*}} .rela.debug_info {
# CHECK:     0x{{[0-9A-F]+}} R_AARCH64_TLS_DTPREL64 var {{.*}}
# CHECK:   }

.section .tdata,"awT",@progbits
.globl var
var:
  .word 0

.section .debug_abbrev,"",@progbits
.byte 1                  // Abbreviation Code
.byte 17                 // DW_TAG_compile_unit
.byte 1                  // DW_CHILDREN_yes
.byte 0                  // EOM(1)
.byte 0                  // EOM(2)

.byte 2                  // Abbreviation Code
.byte 52                 // DW_TAG_variable
.byte 0                  // DW_CHILDREN_no
.byte 2;                 // DW_AT_location
.byte 24                 // DW_FORM_exprloc
.byte 0                  // EOM(1)
.byte 0                  // EOM(2)

.section        .debug_info,"",@progbits
.Lcu_begin0:
  .word .Lcu_end - .Lcu_body // Length of Unit
.Lcu_body:
  .hword 4               // DWARF version number
  .word   .debug_abbrev  // Offset Into Abbrev. Section
  .byte   8              // Address Size (in bytes)
  .byte   1              // Abbrev [1] DW_TAG_compile_unit
  .byte   2              // Abbrev [2] DW_TAG_variable
  .xword  var@DTPREL
.Lcu_end:
