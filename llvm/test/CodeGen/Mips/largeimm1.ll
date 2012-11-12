; RUN: llc -march=mipsel < %s | FileCheck %s

; CHECK: lui ${{[0-9]+}}, 49152
; CHECK: lui ${{[0-9]+}}, 16384
define void @f() nounwind {
entry:
  %a1 = alloca [1073741824 x i8], align 1
  %arrayidx = getelementptr inbounds [1073741824 x i8]* %a1, i32 0, i32 1048676
  call void @f2(i8* %arrayidx) nounwind
  ret void
}

declare void @f2(i8*)
