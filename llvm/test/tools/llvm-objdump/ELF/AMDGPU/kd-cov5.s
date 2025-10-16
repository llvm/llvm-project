; RUN: sed 's/CODE_OBJECT_VERSION/5/g' %s \
; RUN:   | llvm-mc --triple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=-xnack,+wavefrontsize32,-wavefrontsize64 -filetype=obj > %t.o
; RUN: llvm-objdump --disassemble-symbols=kernel.kd %t.o | FileCheck %s --check-prefixes=COV5,CHECK

; RUN: sed 's/CODE_OBJECT_VERSION/4/g' %s \
; RUN:   | llvm-mc --triple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=-xnack,+wavefrontsize32,-wavefrontsize64 -filetype=obj > %t.o
; RUN: llvm-objdump --disassemble-symbols=kernel.kd %t.o | FileCheck %s --check-prefixes=COV4,CHECK

;; Make sure we override the default COV in the disassembler on COV6 (there
;; currently aren't any differences between 5 and 6, so set the default to 4 so
;; we can verify that the default is at least overridden)
; RUN: sed 's/CODE_OBJECT_VERSION/6/g' %s \
; RUN:   | llvm-mc --triple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=-xnack,+wavefrontsize32,-wavefrontsize64 -filetype=obj > %t.o
; RUN: llvm-objdump -mllvm --amdhsa-code-object-version=4 --disassemble-symbols=kernel.kd %t.o | FileCheck %s --check-prefixes=COV5,CHECK

;; Verify that .amdhsa_uses_dynamic_stack is only printed on COV5+.

; CHECK: .amdhsa_kernel kernel
; COV5: .amdhsa_uses_dynamic_stack 0
; COV4-NOT: .amdhsa_uses_dynamic_stack
; CHECK: .end_amdhsa_kernel

.amdhsa_code_object_version CODE_OBJECT_VERSION

.amdhsa_kernel kernel
  .amdhsa_next_free_vgpr 32
  .amdhsa_next_free_sgpr 32
.end_amdhsa_kernel
