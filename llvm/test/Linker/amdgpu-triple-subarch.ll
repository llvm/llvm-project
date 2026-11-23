
; Merge amdgpu + amdgpu9 => amdgpu9 without a warning.
; RUN: llvm-link -S %s %S/Inputs/amdgpu-no-subarch.ll -o %t0.ll 2>&1 | FileCheck --allow-empty -check-prefix=NO-WARN %s
; RUN: llvm-link -S %S/Inputs/amdgpu-no-subarch.ll %s -o %t1.ll 2>&1 | FileCheck --allow-empty -check-prefix=NO-WARN %s
; RUN: FileCheck -check-prefix=LINKED0 %s < %t0.ll
; RUN: FileCheck -check-prefix=LINKED0 %s < %t1.ll

; Invalid merge of amdgpu9 + amdgpu10, but the second triple wins and a warning is produced
; RUN: llvm-link -S %s %S/Inputs/amdgpu10-subarch.ll -o %t2.ll 2>&1 | FileCheck -check-prefix=WARN_A %s
; RUN: llvm-link -S %S/Inputs/amdgpu10-subarch.ll %s -o %t3.ll 2>&1 | FileCheck -check-prefix=WARN_B %s
; RUN: FileCheck -check-prefix=LINKED1 %s < %t2.ll
; RUN: FileCheck -check-prefix=LINKED2 %s < %t3.ll

; Merge amdgpu9 + amdgpu9.00 = amdgpu9.00, and do not warn.
; RUN: llvm-link -S %s %S/Inputs/amdgpu9.00-subarch.ll -o %t4.ll 2>&1 | FileCheck --allow-empty -check-prefix=NO-WARN %s
; RUN: llvm-link -S %S/Inputs/amdgpu9.00-subarch.ll %s -o %t5.ll 2>&1 | FileCheck --allow-empty -check-prefix=NO-WARN %s
; RUN: FileCheck -check-prefix=LINKED3 %s < %t4.ll
; RUN: FileCheck -check-prefix=LINKED3 %s < %t5.ll

; NO-WARN: {{^$}}

; LINKED0: target triple = "amdgpu9-amd-amdhsa"
; LINKED1: target triple = "amdgpu9-amd-amdhsa"
; LINKED2: target triple = "amdgpu10-amd-amdhsa"
; LINKED3: target triple = "amdgpu9.00-amd-amdhsa"

; Check that there is no warning when linking an amdgpu triple without
; a subarch with one with a subarch.


; WARN_A: warning: Linking two modules of different target triples: '{{.*}}' is 'amdgpu10-amd-amdhsa' whereas 'llvm-link' is 'amdgpu9-amd-amdhsa'
; WARN_B: warning: Linking two modules of different target triples: '{{.*}}' is 'amdgpu9-amd-amdhsa' whereas 'llvm-link' is 'amdgpu10-amd-amdhsa'


target triple = "amdgpu9-amd-amdhsa"

declare i32 @foo()

define i32 @bar() {
  %x = call i32 @foo()
  ret i32 %x
}
