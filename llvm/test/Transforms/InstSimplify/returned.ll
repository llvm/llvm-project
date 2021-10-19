; RUN: opt -instsimplify -S < %s | FileCheck %s

define i1 @bitcast() {
; CHECK-LABEL: @bitcast(
  %a = alloca i32
  %b = alloca i64
  %x = bitcast i32* %a to i8*
  %z = bitcast i64* %b to i8*
  %y = call i8* @func1(i8* %z)
  %cmp = icmp eq i8* %x, %y
  ret i1 %cmp
; CHECK-NEXT: ret i1 false
}

%gept = type { i32, i32 }

define i1 @gep3() {
; CHECK-LABEL: @gep3(
  %x = alloca %gept, align 8
  %a = getelementptr %gept, %gept* %x, i64 0, i32 0
  %y = call %gept* @func2(%gept* %x)
  %b = getelementptr %gept, %gept* %y, i64 0, i32 1
  %equal = icmp eq i32* %a, %b
  ret i1 %equal
; CHECK-NEXT: ret i1 false
}

declare i8* @func1(i8* returned) nounwind readnone willreturn
declare %gept* @func2(%gept* returned) nounwind readnone willreturn

