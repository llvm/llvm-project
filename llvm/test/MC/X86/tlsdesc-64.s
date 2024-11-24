# RUN: llvm-mc -triple x86_64-pc-linux-musl %s | FileCheck --check-prefix=PRINT %s

# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-musl %s -o %t
# RUN: llvm-readelf -s %t | FileCheck --check-prefix=SYM %s
# RUN: llvm-objdump -d -r --no-show-raw-insn %t | FileCheck --match-full-lines %s

# PRINT:      leaq a@tlsdesc(%rip), %rax
# PRINT-NEXT: callq *a@tlscall(%rax)

# SYM: TLS GLOBAL DEFAULT UND a

# CHECK:      0: leaq (%rip), %rax  # 0x7 <{{.*}}>
# CHECK-NEXT:   0000000000000003: R_X86_64_GOTPC32_TLSDESC a-0x4
# CHECK-NEXT: 7: callq *(%rax)
# CHECK-NEXT:   0000000000000007: R_X86_64_TLSDESC_CALL a

leaq a@tlsdesc(%rip), %rax
call *a@tlscall(%rax)
addq %fs:0, %rax

# PRINT:      leaq a@tlsdesc(%rip), %r16
# PRINT-NEXT: callq *a@tlscall(%r16)

# CHECK:      12: leaq (%rip), %r16  # 0x1a <{{.*}}>
# CHECK-NEXT:   0000000000000016: R_X86_64_CODE_4_GOTPC32_TLSDESC a-0x4
# CHECK-NEXT: 1a: callq *(%r16)
# CHECK-NEXT:   000000000000001a: R_X86_64_TLSDESC_CALL a

leaq a@tlsdesc(%rip), %r16
call *a@tlscall(%r16)
addq %fs:0, %r16
