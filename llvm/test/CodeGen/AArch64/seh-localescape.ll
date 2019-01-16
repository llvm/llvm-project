; RUN: llc -mtriple arm64-windows %s -o - | FileCheck %s

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @foo() {
entry:
; CHECK-LABEL: foo
; CHECK: .set .Lfoo$frame_escape_0, -4

  %count = alloca i32, align 4
  call void (...) @llvm.localescape(i32* %count)
  ret i32 0
}

define internal i32 @"?filt$0@0@foo@@"(i8* %exception_pointers, i8* %frame_pointer) {
entry:
; CHECK-LABEL: @"?filt$0@0@foo@@"
; CHECK: movz    x8, #:abs_g1_s:.Lfoo$frame_escape_0
; CHECK: movk    x8, #:abs_g0_nc:.Lfoo$frame_escape_0

  %0 = call i8* @llvm.localrecover(i8* bitcast (i32 ()* @foo to i8*), i8* %frame_pointer, i32 0)
  %count = bitcast i8* %0 to i32*
  %1 = load i32, i32* %count, align 4
  ret i32 %1
}

; Function Attrs: nounwind readnone
declare i8* @llvm.localrecover(i8*, i8*, i32) #2

; Function Attrs: nounwind
declare void @llvm.localescape(...) #3
