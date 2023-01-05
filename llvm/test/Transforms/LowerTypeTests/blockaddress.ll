; RUN: opt -S %s -passes=lowertypetests | FileCheck %s


; CHECK: define hidden ptr @f2.cfi() !type !0 {
; CHECK-NEXT:  br label %b
; CHECK: b:
; CHECK-NEXT:  ret ptr blockaddress(@f2.cfi, %b)
; CHECK-NEXT: }

target triple = "x86_64-unknown-linux"

define void @f1() {
entry:
  %0 = call i1 @llvm.type.test(ptr @f2, metadata !"_ZTSFvP3bioE")
  ret void
}

declare i1 @llvm.type.test(ptr, metadata)

define ptr @f2() !type !5 {
  br label %b

b:
  ret ptr blockaddress(@f2, %b)
}

!5 = !{i64 0, !"_ZTSFvP3bioE"}
