# RUN: llvm-mc -triple x86_64-pc-linux-gnux32 %s | FileCheck --check-prefix=PRINT %s

# RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnux32 %s -o %t
# RUN: llvm-readelf -s %t | FileCheck --check-prefix=SYM %s
# RUN: llvm-objdump --no-print-imm-hex -d -r %t | FileCheck --match-full-lines %s

# PRINT:      leal a@tlsdesc(%rip), %eax
# PRINT-NEXT: callq *a@tlscall(%eax)

# SYM: TLS GLOBAL DEFAULT UND a

# CHECK:      0: 40 8d 05 00 00 00 00  leal (%rip), %eax  # 0x7 <{{.*}}>
# CHECK-NEXT:   00000003: R_X86_64_GOTPC32_TLSDESC a-0x4
# CHECK-NEXT: 7: 67 ff 10              callq *(%eax)
# CHECK-NEXT:   00000007: R_X86_64_TLSDESC_CALL a
# CHECK-NEXT: a: 8d 05 34 12 00 00     leal 4660(%rip), %eax # {{.*}}

lea a@tlsdesc(%rip), %eax
call *a@tlscall(%eax)
lea 0x1234(%rip), %eax
