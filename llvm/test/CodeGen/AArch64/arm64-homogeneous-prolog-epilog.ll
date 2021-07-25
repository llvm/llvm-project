; RUN: llc < %s -mtriple=arm64-apple-ios7.0 -homogeneous-prolog-epilog| FileCheck %s
; RUN: llc < %s -mtriple=aarch64-unknown-linux-gnu  -homogeneous-prolog-epilog | FileCheck %s --check-prefixes=CHECK-LINUX

; CHECK-LABEL: __Z3hooii:
; CHECK:      stp     x29, x30, [sp, #-16]!
; CHECK-NEXT: bl      _OUTLINED_FUNCTION_PROLOG_x30x29x19x20x21x22
; CHECK:      bl      __Z3gooi
; CHECK:      bl      __Z3gooi
; CHECK:      bl      _OUTLINED_FUNCTION_EPILOG_x30x29x19x20x21x22
; CHECK-NEXT: b __Z3gooi

; CHECK-LINUX-LABEL: _Z3hooii:
; CHECK-LINUX:      stp     x29, x30, [sp, #-48]!
; CHECK-LINUX-NEXT: bl      OUTLINED_FUNCTION_PROLOG_x19x20x21x22x30x29
; CHECK-LINUX:      bl      _Z3gooi
; CHECK-LINUX:      bl      _Z3gooi
; CHECK-LINUX:      bl      OUTLINED_FUNCTION_EPILOG_x19x20x21x22x30x29
; CHECK-LINUX-NEXT: b _Z3gooi

define i32 @_Z3hooii(i32 %b, i32 %a) nounwind ssp minsize {
  %1 = tail call i32 @_Z3gooi(i32 %b)
  %2 = tail call i32 @_Z3gooi(i32 %a)
  %3 = add i32 %a, %b
  %4 = add i32 %3, %1
  %5 = add i32 %4, %2
  %6 = tail call i32 @_Z3gooi(i32 %5)
  ret i32 %6
}

declare i32 @_Z3gooi(i32);

; CHECK-LABEL: _foo:
; CHECK:        sub sp, sp, #16
; CHECK:        bl  _goo
; CHECK:        add sp, sp, #16

define i32 @foo(i32 %c) nounwind minsize {
entry:
  %buffer = alloca [1 x i32], align 4
  %0 = bitcast [1 x i32]* %buffer to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %0)
  %arraydecay = getelementptr inbounds [1 x i32], [1 x i32]* %buffer, i64 0, i64 0
  %call = call i32 @goo(i32* nonnull %arraydecay)
  %sub = sub nsw i32 %c, %call
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %0)

  ret i32 %sub
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)
declare i32 @goo(i32*)
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)

; CHECK-LABEL: _OUTLINED_FUNCTION_PROLOG_x30x29x19x20x21x22:
; CHECK:      stp     x22, x21, [sp, #-32]!
; CHECK-NEXT: stp     x20, x19, [sp, #16]
; CHECK-NEXT: ret

; CHECK-LABEL: _OUTLINED_FUNCTION_EPILOG_x30x29x19x20x21x22:
; CHECK:      mov     x16, x30
; CHECK-NEXT: ldp     x29, x30, [sp, #32]
; CHECK-NEXT: ldp     x20, x19, [sp, #16]
; CHECK-NEXT: ldp     x22, x21, [sp], #48
; CHECK-NEXT: ret     x16

; CHECK-LINUX-LABEL: OUTLINED_FUNCTION_PROLOG_x19x20x21x22x30x29:
; CHECK-LINUX:      stp     x22, x21, [sp, #16]
; CHECK-LINUX-NEXT: stp     x20, x19, [sp, #32]
; CHECK-LINUX-NEXT: ret

; CHECK-LINUX-LABEL: OUTLINED_FUNCTION_EPILOG_x19x20x21x22x30x29:
; CHECK-LINUX:      mov     x16, x30
; CHECK-LINUX-NEXT: ldp     x20, x19, [sp, #32]
; CHECK-LINUX-NEXT: ldp     x22, x21, [sp, #16]
; CHECK-LINUX-NEXT: ldp     x29, x30, [sp], #48
; CHECK-LINUX-NEXT: ret     x16
