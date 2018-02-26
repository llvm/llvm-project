; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64-- -mcpu=a2 | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- -mcpu=440 | FileCheck %s -check-prefix=BE-CHK

define i32 @has_a_fence(i32 %a, i32 %b) nounwind {
entry:
  fence acquire
  %cond = icmp eq i32 %a, %b
  br i1 %cond, label %IfEqual, label %IfUnequal

IfEqual:
  fence release
; CHECK: sync
; CHECK-NOT: msync
; BE-CHK: msync
  br label %end

IfUnequal:
  fence release
; CHECK: sync
; CHECK-NOT: msync
; BE-CHK: msync
  ret i32 0

end:
  ret i32 1
}

