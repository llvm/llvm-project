; Testcase to check that module with different branch-target-enforcement can
; be mixed.
;
; RUN: llvm-as %s -o %t1.bc
; RUN: llvm-as %p/Inputs/foo.ll -o %t2.bc
; RUN: llvm-lto -exported-symbol main \
; RUN:          -exported-symbol foo \
; RUN:          -filetype=obj \
; RUN:           %t1.bc %t2.bc \
; RUN:           -o %t1.exe 2>&1 | FileCheck --allow-empty %s
; RUN: llvm-objdump -d %t1.exe | FileCheck --check-prefix=CHECK-DUMP %s
; RUN: llvm-readelf -n %t1.exe | FileCheck --check-prefix=CHECK-PROP %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

declare i32 @foo();

define i32 @main() {
entry:
  %add = call i32 @foo()
  ret i32 %add
}

!llvm.module.flags = !{!0, !1, !2, !3 }
!0 = !{i32 8, !"branch-target-enforcement", i32 0}
!1 = !{i32 8, !"sign-return-address", i32 1}
!2 = !{i32 8, !"sign-return-address-all", i32 0}
!3 = !{i32 8, !"sign-return-address-with-bkey", i32 0}

; CHECK-NOT: linking module flags 'branch-target-enforcement': IDs have conflicting values in
; CHECK-DUMP: <main>:
; CHECK-DUMP:      bl      0x8 <main+0x8>
; CHECK-DUMP: <foo>:
; CHECK-DUMP:     paciasp

; `main` doesn't support BTI while `foo` does, so in the binary
; we should see only PAC which is supported by both.
; CHECK-PROP:   Properties: aarch64 feature: PAC