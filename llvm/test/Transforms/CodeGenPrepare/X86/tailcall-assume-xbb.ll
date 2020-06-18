; RUN: opt -codegenprepare -S -mtriple=x86_64-linux < %s | FileCheck %s

; The ret instruction can be duplicated into BB case2 even though there is an
; intermediate BB exit1 and call to llvm.assume.

@ptr = external global i8*, align 8

; CHECK:       %ret1 = tail call i8* @qux()
; CHECK-NEXT:  ret i8* %ret1

; CHECK:       %ret2 = tail call i8* @bar()
; CHECK-NEXT:  ret i8* %ret2

define i8* @foo(i64 %size, i64 %v1, i64 %v2) {
entry:
  %cmp1 = icmp ult i64 %size, 1025
  br i1 %cmp1, label %if.end, label %case1

case1:
  %ret1 = tail call i8* @qux()
  br label %exit2

if.end:
  %cmp2 = icmp ult i64 %v1, %v2
  br i1 %cmp2, label %case3, label %case2

case2:
  %ret2 = tail call i8* @bar()
  br label %exit1

case3:
  %ret3 = load i8*, i8** @ptr, align 8
  br label %exit1

exit1:
  %retval1 = phi i8* [ %ret2, %case2 ], [ %ret3, %case3 ]
  %cmp3 = icmp ne i8* %retval1, null
  tail call void @llvm.assume(i1 %cmp3)
  br label %exit2

exit2:
  %retval2 = phi i8* [ %ret1, %case1 ], [ %retval1, %exit1 ]
  ret i8* %retval2
}

declare void @llvm.assume(i1)
declare i8* @qux()
declare i8* @bar()
