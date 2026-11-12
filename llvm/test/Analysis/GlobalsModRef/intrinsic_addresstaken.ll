; RUN: opt -aa-pipeline=globals-aa -passes='require<globals-aa>,gvn' -S < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@deallocCalled = internal global i8 0, align 1
@.objc_method_list = internal global { ptr } { ptr  @_i_Associated__dealloc }, align 8
@._OBJC_CLASS_Associated = global { ptr } { ptr @.objc_method_list }, align 8
@._OBJC_INIT_CLASS_Associated = global ptr @._OBJC_CLASS_Associated
@llvm.used = appending global [1 x ptr] [ptr @._OBJC_INIT_CLASS_Associated]

define internal ptr @_i_Associated__dealloc() {
entry:
  store i8 1, ptr @deallocCalled, align 1
  ret ptr null
}

; CHECK-LABEL: @main()
define dso_local i32 @main() {
entry:
  %tmp0 = call ptr @llvm.stacksave() #1
  %tmp6 = load i8, ptr @deallocCalled, align 1
  %tobool = icmp ne i8 %tmp6, 0
  br i1 %tobool, label %if.else, label %if.end

if.else:                                          ; preds = %entry
  call void @__assert_fail() #0
  unreachable

; CHECK-LABEL: if.end:
; CHECK-NEXT: call void @llvm.stackrestore
if.end:                                           ; preds = %entry
  call void @llvm.stackrestore(ptr %tmp0)
  %tmp7 = load i8, ptr @deallocCalled, align 1
  %tobool3 = icmp ne i8 %tmp7, 0
  br i1 %tobool3, label %if.end6, label %if.else5

if.else5:                                         ; preds = %if.end
  call void @__assert_fail() #0
  unreachable

if.end6:                                          ; preds = %if.end
  store i8 0, ptr @deallocCalled, align 1
  ret i32 0
}

declare ptr @llvm.stacksave() #1
declare void @llvm.stackrestore(ptr) #1
declare dso_local void @__assert_fail() #0

attributes #0 = { noreturn nounwind }
attributes #1 = { nounwind }
