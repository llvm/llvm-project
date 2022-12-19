; Stop after bbsections-prepare and check MIR output for section type.
; RUN: echo '!_Z3foob' > %t
; RUN: echo '!!1' >> %t
; RUN: echo '!!2' >> %t
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t -stop-after=bbsections-prepare | FileCheck %s -check-prefix=CHECK

@_ZTIb = external constant i8*
define dso_local i32 @_Z3foob(i1 zeroext %0) {
  %2 = alloca i32, align 4
  %3 = alloca i8, align 1
  %4 = zext i1 %0 to i8
  store i8 %4, i8* %3, align 1
  %5 = load i8, i8* %3, align 1
  %6 = trunc i8 %5 to i1
  br i1 %6, label %7, label %8

7:                                                ; preds = %1
  store i32 1, i32* %2, align 4
  br label %9

8:                                                ; preds = %1
  store i32 0, i32* %2, align 4
  br label %9

9:                                                ; preds = %8, %7
  %10 = load i32, i32* %2, align 4
  ret i32 %10
}

; CHECK: bb.0 (%ir-block.1, bbsections Cold):
; CHECK: bb.3 (%ir-block.9, bbsections Cold):
; CHECK: bb.1 (%ir-block.7)
; CHECK: bb.2 (%ir-block.8, bbsections 1):
