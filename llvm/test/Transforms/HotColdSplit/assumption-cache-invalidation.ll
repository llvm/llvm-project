; REQUIRES: asserts
; RUN: opt -S -instsimplify -hotcoldsplit -hotcoldsplit-threshold=-1 -debug < %s 2>&1 | FileCheck %s
; RUN: opt -instcombine -hotcoldsplit -instsimplify %s -o /dev/null

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

%a = type { i64, i64 }
%b = type { i64 }

; CHECK: @f
; CHECK-LABEL: codeRepl:
; CHECK-NOT: @llvm.assume
; CHECK: }
; CHECK: declare {{.*}}@llvm.assume
; CHECK: define {{.*}}@f.cold.1()
; CHECK-LABEL: newFuncRoot:
; CHECK: }
; CHECK: define {{.*}}@f.cold.2(i64 %0)
; CHECK-LABEL: newFuncRoot:
; CHECK: %1 = icmp eq i64 %0, 0
; CHECK-NOT: call void @llvm.assume

define void @f() {
entry:
  %0 = getelementptr inbounds %a, %a* null, i64 0, i32 1
  br label %label

label:
  %1 = bitcast i64* %0 to %b**
  %2 = load %b*, %b** %1, align 8
  %3 = getelementptr inbounds %b, %b* %2, i64 undef, i32 0
  %4 = load i64, i64* %3, align 8
  %5 = icmp ugt i64 %4, 1
  br i1 %5, label %if.then, label %if.else

if.then:
  unreachable

if.else:
  call void @g(i8* undef)
  %6 = load i64, i64* undef, align 8
  %7 = and i64 %6, -16
  %8 = inttoptr i64 %7 to i8*
  %9 = icmp eq i64 %4, 0
  call void @llvm.assume(i1 %9)
  unreachable
}

declare void @g(i8*)

declare void @llvm.assume(i1)

