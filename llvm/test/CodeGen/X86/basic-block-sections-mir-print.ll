; Stop after bbsections-prepare and check MIR output for section type.
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-address-map -stop-after=bbsections-prepare | FileCheck %s -check-prefix=BBADDRMAP
; RUN: echo '!_Z3foob' > %t
; RUN: echo '!!1' >> %t
; RUN: echo '!!2' >> %t
; RUN: llc < %s -O0 -mtriple=x86_64-pc-linux -function-sections -basic-block-sections=%t -stop-after=bbsections-prepare | FileCheck %s -check-prefix=BBSECTIONS

@_ZTIb = external constant ptr
define dso_local i32 @_Z3foob(i1 zeroext %0) {
  %2 = alloca i32, align 4
  %3 = alloca i8, align 1
  %4 = zext i1 %0 to i8
  store i8 %4, ptr %3, align 1
  %5 = load i8, ptr %3, align 1
  %6 = trunc i8 %5 to i1
  br i1 %6, label %7, label %8

7:                                                ; preds = %1
  store i32 1, ptr %2, align 4
  br label %9

8:                                                ; preds = %1
  store i32 0, ptr %2, align 4
  br label %9

9:                                                ; preds = %8, %7
  %10 = load i32, ptr %2, align 4
  ret i32 %10
}

; BBSECTIONS: bb.0 (%ir-block.1, bbsections Cold, bb_id 0):
; BBSECTIONS: bb.3 (%ir-block.9, bbsections Cold, bb_id 3):
; BBSECTIONS: bb.1 (%ir-block.7, bb_id 1)
; BBSECTIONS: bb.2 (%ir-block.8, bbsections 1, bb_id 2):

; BBADDRMAP: bb.0 (%ir-block.1, bb_id 0):
; BBADDRMAP: bb.1 (%ir-block.7, bb_id 1):
; BBADDRMAP: bb.2 (%ir-block.8, bb_id 2):
; BBADDRMAP: bb.3 (%ir-block.9, bb_id 3):
