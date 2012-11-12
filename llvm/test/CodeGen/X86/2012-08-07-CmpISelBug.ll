; RUN: llc < %s -mtriple=x86_64-apple-macosx | FileCheck %s

; Cmp lowering should not look past the truncate unless the high bits are known
; zero.
; rdar://12027825

define void @foo(i8 %arg4, i32 %arg5, i32* %arg14) nounwind {
bb:
; CHECK: foo:
; CHECK-NOT: testl
; CHECK: testb
  %tmp48 = zext i8 %arg4 to i32
  %tmp49 = and i32 %tmp48, 32
  %tmp50 = add i32 %tmp49, 1593371643
  %tmp55 = sub i32 %tmp50, 0
  %tmp56 = add i32 %tmp55, 7787538
  %tmp57 = xor i32 %tmp56, 1601159181
  %tmp58 = xor i32 %arg5, 1601159181
  %tmp59 = and i32 %tmp57, %tmp58
  %tmp60 = add i32 %tmp59, -1263900958
  %tmp67 = sub i32 %tmp60, 0
  %tmp103 = xor i32 %tmp56, 13
  %tmp104 = trunc i32 %tmp103 to i8
  %tmp105 = sub i8 0, %tmp104
  %tmp106 = add i8 %tmp105, -103
  %tmp113 = sub i8 %tmp106, 0
  %tmp114 = add i8 %tmp113, -72
  %tmp141 = icmp ne i32 %tmp67, -1263900958
  %tmp142 = select i1 %tmp141, i8 %tmp114, i8 undef
  %tmp143 = xor i8 %tmp142, 81
  %tmp144 = zext i8 %tmp143 to i32
  %tmp145 = add i32 %tmp144, 2062143348
  %tmp152 = sub i32 %tmp145, 0
  store i32 %tmp152, i32* %arg14
  ret void
}
