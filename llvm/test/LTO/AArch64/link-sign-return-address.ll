; Testcase to check that module with different sign return address can
; be mixed.
;
; RUN: llvm-as %s -o %t1.bc
; RUN: llvm-as %p/Inputs/foo.ll -o %t2.bc
; RUN: llvm-as %p/Inputs/bar.ll -o %t3.bc
; RUN: llvm-lto -exported-symbol main \
; RUN:          -exported-symbol foo \
; RUN:          -filetype=obj \
; RUN:           %t3.bc %t2.bc %t1.bc \
; RUN:           -o %t1.exe 2>&1
; RUN: llvm-objdump -d %t1.exe | FileCheck --check-prefix=CHECK-DUMP %s
; RUN: llvm-readelf -n %t1.exe | FileCheck --allow-empty --check-prefix=CHECK-PROP %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

declare i32 @foo();
declare void @baz();
declare void @bar();

define i32 @main() {
entry:
  %add = call i32 @foo()
  call void @bar()
  call void @baz()
  ret i32 %add
}

!llvm.module.flags = !{!0, !1, !2, !3 }
!0 = !{i32 8, !"branch-target-enforcement", i32 0}
!1 = !{i32 8, !"sign-return-address", i32 0}
!2 = !{i32 8, !"sign-return-address-all", i32 0}
!3 = !{i32 8, !"sign-return-address-with-bkey", i32 0}


; CHECK-DUMP: <bar>:
; CHECK-DUMP:     ret
; CHECK-DUMP: <baz>:
; CHECK-DUMP:     bti     c
; CHECK-DUMP:     ret
; CHECK-DUMP: <foo>:
; CHECK-DUMP:     pacibsp
; CHECK-DUMP:     mov     w0, #0x2a
; CHECK-DUMP:     autibsp
; CHECK-DUMP:     ret
; CHECK-DUMP: <main>:
; CHECK-DUMP-NOT:  paciasp
; CHECK-DUMP:      str     x30,
; CHECK-DUMP:      bl      0x20 <main+0x4>
; CHECK-DUMP:      bl      0x0 <bar>
; CHECK-DUMP:      bl      0x4 <baz>

; `main` doesn't support PAC sign-return-address while `foo` does, so in the binary
; we should not see anything.
; CHECK-PROP-NOT:   Properties: aarch64 feature: PAC