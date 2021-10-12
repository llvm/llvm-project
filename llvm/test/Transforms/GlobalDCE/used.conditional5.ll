; RUN: opt -S -globaldce < %s | FileCheck %s

@target = internal unnamed_addr constant i32 46
@dep1 = internal unnamed_addr constant i32 732

@llvm.used = appending global [1 x i8*] [
  i8* bitcast (i32* @target to i8*)
], section "llvm.metadata"

!0 = !{i32* @target, i32 0, !{i8* bitcast (i32* @dep1 to i8*)}}
!llvm.used.conditional = !{!0}

; CHECK-NOT: @target
; CHECK-NOT: @dep1
; CHECK: !llvm.used.conditional = !{!0}
; CHECK: !0 = distinct !{null, i32 0, !1}
; CHECK: !1 = !{i8* undef}
