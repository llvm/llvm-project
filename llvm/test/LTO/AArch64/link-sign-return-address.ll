;; Testcase to check that module with different sign return address can
;; be mixed.
;
; RUN: llvm-as %s -o %t1.bc
; RUN: llvm-as %p/Inputs/foo.ll -o %t2.bc
; RUN: llvm-as %p/Inputs/fiz.ll -o %t3.bc
; RUN: llvm-as %p/Inputs/bar.ll -o %t4.bc
; RUN: llvm-as %p/Inputs/old.ll -o %t5.bc
; RUN: llvm-lto -exported-symbol main \
; RUN:          -exported-symbol foo_on \
; RUN:          -exported-symbol foo_off \
; RUN:          -exported-symbol fiz_on \
; RUN:          -exported-symbol fiz_off \
; RUN:          -exported-symbol bar \
; RUN:          -exported-symbol baz \
; RUN:          -exported-symbol old_bti \
; RUN:          -exported-symbol old_pac \
; RUN:          -exported-symbol old_none \
; RUN:          -filetype=obj \
; RUN:          %t5.bc %t4.bc %t3.bc %t2.bc %t1.bc \
; RUN:           -o %t1.exe 2>&1
; RUN: llvm-objdump -d %t1.exe | FileCheck --check-prefix=CHECK-DUMP %s
; RUN: llvm-readelf -n %t1.exe | FileCheck --allow-empty --check-prefix=CHECK-PROP %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

declare i32 @foo_on();
declare i32 @foo_off();
declare i32 @fiz_on();
declare i32 @fiz_off();
declare void @baz();
declare void @bar();
declare i32 @old_bti();
declare i32 @old_pac();
declare i32 @old_none();

define i32 @main() #0 {
entry:
  call i32 @foo_on()
  call i32 @foo_off()
  call i32 @fiz_on()
  call i32 @fiz_off()
  call void @bar()
  call void @baz()
  call i32 @old_bti()
  call i32 @old_pac()
  call i32 @old_none()
  ret i32 0
}

attributes #0 = { noinline nounwind optnone }

!llvm.module.flags = !{!0, !1, !2, !3 }
!0 = !{i32 8, !"branch-target-enforcement", i32 0}
!1 = !{i32 8, !"sign-return-address", i32 0}
!2 = !{i32 8, !"sign-return-address-all", i32 0}
!3 = !{i32 8, !"sign-return-address-with-bkey", i32 0}


; CHECK-DUMP-LABEL: <old_bti>:
; CHECK-DUMP-NEXT:     bti c
; CHECK-DUMP-NEXT:     mov     w0, #0x2
; CHECK-DUMP-NEXT:     ret

; CHECK-DUMP-LABEL: <old_pac>:
; CHECK-DUMP-NEXT:     paciasp
; CHECK-DUMP-NEXT:     mov     w0, #0x2
; CHECK-DUMP-NEXT:     autiasp
; CHECK-DUMP-NEXT:     ret

; CHECK-DUMP-LABEL: <old_none>:
; CHECK-DUMP-NEXT:     mov     w0, #0x3
; CHECK-DUMP-NEXT:     ret

; CHECK-DUMP-LABEL: <bar>:
; CHECK-DUMP-NEXT:     ret

; CHECK-DUMP-LABEL: <baz>:
; CHECK-DUMP-NEXT:     bti c
; CHECK-DUMP-NEXT:     ret

;; fiz.ll represents a module with the old style of the function attributes.
;; fiz_on shall have PAC with A-key as it requested at module level.
; CHECK-DUMP-LABEL: <fiz_on>:
; CHECK-DUMP-NEXT:     paciasp
; CHECK-DUMP-NEXT:     str     x30, [sp, #-0x10]!
; CHECK-DUMP-NEXT:     bl      0x38 <fiz_on+0x8>
; CHECK-DUMP-NEXT:     mov     w0, #0x2a
; CHECK-DUMP-NEXT:     ldr     x30, [sp], #0x10
; CHECK-DUMP-NEXT:     autiasp
; CHECK-DUMP-NEXT:     ret

;; fiz_off shall not have BTI or PAC instructions as they are disabled at function scope.
; CHECK-DUMP-LABEL:  <fiz_off>:
; CHECK-DUMP-NEXT:       mov     w0, #0x2b
; CHECK-DUMP-NEXT:       ret

;; foo.ll represents a module with the old style of the function attributes.
;; foo_on shall have PAC with B-key as it requested at module level.
; CHECK-DUMP-LABEL: <foo_on>:
; CHECK-DUMP-NEXT:     pacibsp
; CHECK-DUMP-NEXT:     mov     w0, #0x2a
; CHECK-DUMP-NEXT:     autibsp
; CHECK-DUMP-NEXT:     ret

;; foo_off shall not have BTI or PAC instructions as they are disabled at function scope.
; CHECK-DUMP-LABEL:  <foo_off>:
; CHECK-DUMP-NEXT:       mov     w0, #0x2b
; CHECK-DUMP-NEXT:       ret

; CHECK-DUMP-LABEL: <main>:
; CHECK-DUMP-NOT:       paciasp
; CHECK-DUMP-NEXT:      str     x30,
; CHECK-DUMP-NEXT:      bl
; CHECK-DUMP-NEXT:      bl
; CHECK-DUMP-NEXT:      bl
; CHECK-DUMP-NEXT:      bl
; CHECK-DUMP-NEXT:      bl
; CHECK-DUMP-NEXT:      bl
; CHECK-DUMP-NEXT:      bl
; CHECK-DUMP-NEXT:      bl
; CHECK-DUMP-NEXT:      bl

;; `main` doesn't support PAC sign-return-address while `foo` does, so in the binary
;; we should not see anything.
; CHECK-PROP-NOT:   Properties: aarch64 feature: PAC
