; RUN: llc -mtriple=x86_64-- -no-integrated-as < %s | FileCheck %s
; PR3701

define i64 @t(ptr %arg) nounwind {
        br i1 true, label %1, label %5

; <label>:1             ; preds = %0
        %2 = icmp eq ptr null, %arg            ; <i1> [#uses=1]
        %3 = tail call ptr asm sideeffect "movl %fs:0,$0", "=r,~{dirflag},~{fpsr},~{flags}"() nounwind         ; <ptr> [#uses=0]
; CHECK: test
; CHECK-NEXT: j
        br i1 %2, label %4, label %5

; <label>:4             ; preds = %1
        ret i64 1

; <label>:5             ; preds = %1
        ret i64 0
}

; Make sure that we translate this to the bswap intrinsic which lowers down without the
; inline assembly.
; CHECK-NOT: #APP
define i32 @s(i32 %argc, ptr nocapture %argv) unnamed_addr nounwind {
entry:
  %0 = trunc i32 %argc to i16
  %asmtmp = tail call i16 asm "rorw $$8, ${0:w}", "=r,0,~{fpsr},~{flags},~{cc}"(i16 %0) nounwind, !srcloc !0
  %1 = zext i16 %asmtmp to i32
  ret i32 %1
}

!0 = !{i64 935930}
