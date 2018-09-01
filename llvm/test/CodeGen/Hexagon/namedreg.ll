; RUN: llc -march=hexagon < %s | FileCheck %s
define dso_local i32 @r19f() #0 {
entry:
  %0 = call i32 @llvm.read_register.i32(metadata !0)
  ret i32 %0
}

; Function Attrs: noinline nounwind optnone
define dso_local i32 @spf() #0 {
entry:
  %0 = call i32 @llvm.read_register.i32(metadata !1)
  ret i32 %0
}

declare i32 @llvm.read_register.i32(metadata) #1

!llvm.named.register.r19 = !{!0}
!llvm.named.register.sp = !{!1}

!0 = !{!"r19"}
!1 = !{!"sp"}
; CHECK: r0 = r19
; CHECK: r0 = r29
