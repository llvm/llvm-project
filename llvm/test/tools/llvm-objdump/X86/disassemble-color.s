# UNSUPPORTED: system-windows
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-objdump -d --no-show-raw-insn --disassembler-color=on %t | FileCheck %s --check-prefix=ATT
# RUN: llvm-objdump -d --no-show-raw-insn --disassembler-color=on -M intel %t | FileCheck %s --check-prefix=INTEL

# ATT:      <.text>:
# ATT-NEXT:  leaq	[0;32m([0;36m%rdx[0;32m,[0;36m%rax[0;32m,[0;31m4[0;32m)[0m, [0;36m%rbx[0m
# ATT-NEXT:  movq	[0;32m(,[0;36m%rax[0;32m)[0m, [0;36m%rbx[0m
# ATT-NEXT:  leaq	[0;32m0x3([0;36m%rdx[0;32m,[0;36m%rax[0;32m)[0m, [0;36m%rbx[0m
# ATT-NEXT:  movq	[0;31m$0x3[0m, [0;36m%rax[0m

# INTEL:      <.text>:
# INTEL-NEXT:  lea	[0;36mrbx[0m, [0;32m[[0;36mrdx[0;32m + 4*[0;36mrax[0;32m][0m
# INTEL-NEXT:  mov	[0;36mrbx[0m, qword ptr [0;32m[1*[0;36mrax[0;32m][0m
# INTEL-NEXT:  lea	[0;36mrbx[0m, [0;32m[[0;36mrdx[0;32m + [0;36mrax[0;32m + [0;31m0x3[0;32m][0m
# INTEL-NEXT:  mov	[0;36mrax[0m, [0;31m0x3[0m

leaq (%rdx,%rax,4), %rbx
movq (,%rax), %rbx
leaq 3(%rdx,%rax), %rbx
movq $3, %rax
