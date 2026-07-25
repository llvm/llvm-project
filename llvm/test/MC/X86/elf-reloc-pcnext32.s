# RUN: llvm-mc -filetype=obj -triple=x86_64 %s | llvm-readobj -r - | FileCheck %s

# CHECK:      Section ({{.*}}) .rela.text {
# CHECK-NEXT:   0x3 R_X86_64_PCNEXT32 target 0xFFFFFFFFFFFFFFFC
# CHECK-NEXT:   0x7 R_X86_64_PCNEXT32 target 0x0
# CHECK-NEXT: }

prefetchit1 target@PCNEXT32(%rip)

.reloc ., R_X86_64_PCNEXT32, target
.long 0
