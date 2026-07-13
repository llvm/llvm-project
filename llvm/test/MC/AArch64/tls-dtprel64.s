// RUN: llvm-mc -triple=aarch64 -show-encoding < %s | FileCheck %s
// RUN: llvm-mc -filetype=obj -triple=aarch64 %s | llvm-readobj -r - | FileCheck --check-prefix=CHECK-ELF %s

# CHECK: .xword %dtprel(var)
# CHECK: .xword	%dtprel(var+1)
# CHECK: .xword	%dtprel(.tdata)
# CHECK: .xword	%dtprel(.tdata+1)

# CHECK-ELF: Relocations [
# CHECK-ELF:   Section (5) .rela.debug_info {
# CHECK-ELF:     0x0 R_AARCH64_TLS_DTPREL64 var 0x0
# CHECK-ELF:     0x8 R_AARCH64_TLS_DTPREL64 var 0x1
# CHECK-ELF:     0x10 R_AARCH64_TLS_DTPREL64 .tdata 0x0
# CHECK-ELF:     0x18 R_AARCH64_TLS_DTPREL64 .tdata 0x1
# CHECK-ELF:   }

.section .tdata,"awT",@progbits
.skip 8
.globl var
var:
  .word 0

.section        .debug_info,"",@progbits
  .xword  %dtprel(var)
  .xword  %dtprel(var+1)
  .xword  %dtprel(.tdata)
  .xword  %dtprel(.tdata+1)
