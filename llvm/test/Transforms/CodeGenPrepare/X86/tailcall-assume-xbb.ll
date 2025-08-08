; RUN: opt -passes='require<profile-summary>,function(codegenprepare)' -S -mtriple=x86_64-linux < %s | FileCheck %s

; The ret instruction can be duplicated into BB case2 even though there is an
; intermediate BB exit1 and call to llvm.assume.

@ptr = external global ptr, align 8

; CHECK:       %ret1 = tail call ptr @qux()
; CHECK-NEXT:  ret ptr %ret1

; CHECK:       %ret2 = tail call ptr @bar()
; CHECK-NEXT:  ret ptr %ret2

define ptr @foo(i64 %size, i64 %v1, i64 %v2) {
entry:
  %a = alloca i8
  call void @llvm.lifetime.start.p0(ptr %a) nounwind
  %cmp1 = icmp ult i64 %size, 1025
  br i1 %cmp1, label %if.end, label %case1

case1:
  %ret1 = tail call ptr @qux()
  br label %exit2

if.end:
  %cmp2 = icmp ult i64 %v1, %v2
  br i1 %cmp2, label %case3, label %case2

case2:
  %ret2 = tail call ptr @bar()
  br label %exit1

case3:
  %ret3 = load ptr, ptr @ptr, align 8
  br label %exit1

exit1:
  %retval1 = phi ptr [ %ret2, %case2 ], [ %ret3, %case3 ]
  %cmp3 = icmp ne ptr %retval1, null
  tail call void @llvm.assume(i1 %cmp3)
  br label %exit2

exit2:
  %retval2 = phi ptr [ %ret1, %case1 ], [ %retval1, %exit1 ]
  call void @llvm.lifetime.end.p0(ptr %a) nounwind
  ret ptr %retval2
}

declare void @llvm.assume(i1)
declare ptr @qux()
declare ptr @bar()
declare void @llvm.lifetime.start.p0(ptr nocapture) nounwind
declare void @llvm.lifetime.end.p0(ptr nocapture) nounwind
