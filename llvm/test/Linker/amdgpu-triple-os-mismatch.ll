; RUN: llvm-link -S %s %S/Inputs/amdgpu-amdpal-no-subarch.ll -o %t0.ll 2>&1 | FileCheck -check-prefix=WARN_A %s
; RUN: llvm-link -S %S/Inputs/amdgpu-amdpal-no-subarch.ll %s -o %t1.ll 2>&1 | FileCheck -check-prefix=WARN_B %s
; RUN: FileCheck -check-prefix=LINKED0 %s < %t0.ll
; RUN: FileCheck -check-prefix=LINKED1 %s < %t1.ll

; WARN_A: warning: Linking two modules of different target triples: '{{.*}}' is 'amdgpu-amd-amdpal' whereas 'llvm-link' is 'amdgpu9-amd-amdhsa'
; WARN_B: warning: Linking two modules of different target triples: '{{.*}}' is 'amdgpu9-amd-amdhsa' whereas 'llvm-link' is 'amdgpu-amd-amdpal'

; LINKED0: target triple = "amdgpu9-amd-amdhsa"
; LINKED1: target triple = "amdgpu-amd-amdpal"

target triple = "amdgpu9-amd-amdhsa"

declare i32 @foo()

define i32 @bar() {
  %x = call i32 @foo()
  ret i32 %x
}
