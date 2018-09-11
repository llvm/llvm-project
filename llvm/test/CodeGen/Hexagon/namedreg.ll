; RUN: llc -mattr=+reserved-r19 -march=hexagon < %s | FileCheck %s
define dso_local i32 @r19f() #0 {
entry:
  %0 = call i32 @llvm.read_register.i32(metadata !0)
  ret i32 %0
}

declare i32 @llvm.read_register.i32(metadata) #1

!llvm.named.register.r19 = !{!0}

!0 = !{!"r19"}
; CHECK: r0 = r19
