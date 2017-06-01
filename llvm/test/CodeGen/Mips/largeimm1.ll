; RUN: llc -march=mipsel -relocation-model=pic < %s | FileCheck %s

define void @f() nounwind {
entry:
  %a1 = alloca [1073741824 x i8], align 1
  %arrayidx = getelementptr inbounds [1073741824 x i8], [1073741824 x i8]* %a1, i32 0, i32 1048676
  call void @f2(i8* %arrayidx) nounwind
  ret void
; CHECK-LABEL: f:

; CHECK: lui    $[[R0:[a-z0-9]+]], 16384
; CHECK: addiu  $[[R1:[a-z0-9]+]], $[[R0]], 24
; CHECK: subu   $sp, $sp, $[[R1]]

; CHECK: lui    $[[R2:[a-z0-9]+]], 16384
; CHECK: addu   ${{[0-9]+}}, $sp, $[[R2]]
}

declare void @f2(i8*)
