; REQUIRES: asserts
; RUN: opt -S -passes='function(instsimplify),hotcoldsplit' -hotcoldsplit-threshold=-1 -debug < %s 2>&1 | FileCheck %s
; RUN: opt -passes='function(instcombine),hotcoldsplit,function(instsimplify)' %s -o /dev/null

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

%a = type { i64, i64 }
%b = type { i64 }

; CHECK: @f
; CHECK-LABEL: codeRepl:
; CHECK-NOT: @llvm.assume
; CHECK: }
; CHECK: declare {{.*}}@llvm.assume
; CHECK: define {{.*}}@f.cold.1(i64 %load1)
; CHECK-LABEL: newFuncRoot:
; CHECK: %cmp1 = icmp eq i64 %load1, 0
; CHECK-NOT: call void @llvm.assume
; CHECK: define {{.*}}@f.cold.2()
; CHECK-LABEL: newFuncRoot:
; CHECK: }

define void @f() {
entry:
  %i = getelementptr inbounds %a, ptr null, i64 0, i32 1
  br label %label

label:                                            ; preds = %entry
  %load0 = load ptr, ptr %i, align 8
  %i3 = getelementptr inbounds %b, ptr %load0, i64 undef, i32 0
  %load1 = load i64, ptr %i3, align 8
  %cmp0 = icmp ugt i64 %load1, 1
  br i1 %cmp0, label %if.then, label %if.else

if.then:                                          ; preds = %label
  unreachable

if.else:                                          ; preds = %label
  call void @g(ptr undef)
  %load2 = load i64, ptr undef, align 8
  %i7 = and i64 %load2, -16
  %i8 = inttoptr i64 %i7 to ptr
  %cmp1 = icmp eq i64 %load1, 0
  call void @llvm.assume(i1 %cmp1)
  unreachable
}

declare void @g(ptr)

declare void @llvm.assume(i1 noundef) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
