; RUN: opt -passes=hotcoldsplit -hotcoldsplit-threshold=0 -S < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

declare void @_Z10sideeffectv()
declare void @llvm.ubsantrap(i8 immarg) cold noreturn nounwind

; Don't outline -fbounds-safety traps

; CHECK-LABEL: define {{.*}}@foo(
; CHECK-NOT: foo.cold.1
define void @foo(i32, ptr) {
  %3 = icmp eq i32 %0, 0
  tail call void @_Z10sideeffectv()
  br i1 %3, label %5, label %4

; <label>:4:                                      ; preds = %2
  tail call void @llvm.ubsantrap(i8 25)
  unreachable

; <label>:5:                                      ; preds = %2
  ret void
}

; Check IR from -funique-traps
;
; CHECK-LABEL: define {{.*}}@bar(
; CHECK-NOT: bar.cold.1
define void @bar(i32, ptr) {
  %3 = icmp eq i32 %0, 0
  tail call void @_Z10sideeffectv()
  br i1 %3, label %5, label %4

; <label>:4:                                      ; preds = %2
  tail call void asm sideeffect "", "n"(i64 0) nounwind
  tail call void @llvm.ubsantrap(i8 25)
  unreachable

; <label>:5:                                      ; preds = %2
  ret void
}


