; RUN: opt -S %s -lowertypetests | FileCheck %s


; CHECK: define internal i8* @f2.cfi() !type !0 {
; CHECK-NEXT:  br label %b
; CHECK: b:
; CHECK-NEXT:  ret i8* blockaddress(@f2.cfi, %b)
; CHECK-NEXT: }

target triple = "x86_64-unknown-linux"

define void @f1() {
entry:
  %0 = call i1 @llvm.type.test(i8* bitcast (i8* ()* @f2 to i8*), metadata !"_ZTSFvP3bioE")
  ret void
}

declare i1 @llvm.type.test(i8*, metadata)

define i8* @f2() !type !5 {
  br label %b

b:
  ret i8* blockaddress(@f2, %b)
}

!5 = !{i64 0, !"_ZTSFvP3bioE"}
