# RUN: llvm-mc -triple i386-pc-linux-musl %s | FileCheck --check-prefix=PRINT %s

# RUN: llvm-mc -filetype=obj -triple i386-pc-linux-musl %s -o %t
# RUN: llvm-readelf -s %t | FileCheck --check-prefix=SYM %s
# RUN: llvm-objdump -d -r --no-show-raw-insn %t | FileCheck %s

# PRINT:      leal a@tlsdesc(%ebx), %eax
# PRINT-NEXT: calll *a@tlscall(%eax)

# SYM: TLS GLOBAL DEFAULT UND a

# CHECK:      0: leal (%ebx), %eax
# CHECK-NEXT:   00000002: R_386_TLS_GOTDESC a
# CHECK-NEXT: 6: calll *(%eax)
# CHECK-NEXT:   00000006: R_386_TLS_DESC_CALL a

leal a@tlsdesc(%ebx), %eax
call *a@tlscall(%eax)
addl %gs:0, %eax
