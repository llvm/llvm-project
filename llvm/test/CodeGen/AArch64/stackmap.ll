; RUN: llc < %s -mtriple=arm64-linux-gnu | FileCheck %s
;
; Note: Print verbose stackmaps using -debug-only=stackmaps.

; CHECK-LABEL:  .section  .llvm_stackmaps
; CHECK-NEXT:  __LLVM_StackMaps:
; Header
; CHECK-NEXT:   .byte 3
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .hword 0
; Num Functions
; CHECK-NEXT:   .word 14
; Num LargeConstants
; CHECK-NEXT:   .word 4
; Num Callsites
; CHECK-NEXT:   .word 18

; Functions and stack size
; CHECK-NEXT:   .xword constantargs
; CHECK-NEXT:   .xword 16
; CHECK-NEXT:   .xword 1
; CHECK-NEXT:   .xword osrinline
; CHECK-NEXT:   .xword 32
; CHECK-NEXT:   .xword 1
; CHECK-NEXT:   .xword osrcold
; CHECK-NEXT:   .xword 16
; CHECK-NEXT:   .xword 1
; CHECK-NEXT:   .xword propertyRead
; CHECK-NEXT:   .xword 16
; CHECK-NEXT:   .xword 1
; CHECK-NEXT:   .xword propertyWrite
; CHECK-NEXT:   .xword 16
; CHECK-NEXT:   .xword 1
; CHECK-NEXT:   .xword jsVoidCall
; CHECK-NEXT:   .xword 16
; CHECK-NEXT:   .xword 1
; CHECK-NEXT:   .xword jsIntCall
; CHECK-NEXT:   .xword 16
; CHECK-NEXT:   .xword 1
; CHECK-NEXT:   .xword spilledValue
; CHECK-NEXT:   .xword 144
; CHECK-NEXT:   .xword 1
; CHECK-NEXT:   .xword spilledStackMapValue
; CHECK-NEXT:   .xword 128
; CHECK-NEXT:   .xword 1
; CHECK-NEXT:   .xword liveConstant
; CHECK-NEXT:   .xword 16
; CHECK-NEXT:   .xword 1
; CHECK-NEXT:   .xword directFrameIdx
; CHECK-NEXT:   .xword 64
; CHECK-NEXT:   .xword 2
; CHECK-NEXT:   .xword longid
; CHECK-NEXT:   .xword 16
; CHECK-NEXT:   .xword 4
; CHECK-NEXT:   .xword clobberLR
; CHECK-NEXT:   .xword 112
; CHECK-NEXT:   .xword 1
; CHECK-NEXT:   .xword needsStackRealignment
; CHECK-NEXT:   .xword -1
; CHECK-NEXT:   .xword 1

; Large Constants
; CHECK-NEXT:   .xword   2147483648
; CHECK-NEXT:   .xword   4294967295
; CHECK-NEXT:   .xword   4294967296
; CHECK-NEXT:   .xword   4294967297

; Callsites
; Constant arguments
;
; CHECK-NEXT:   .xword   1
; CHECK-NEXT:   .word   .L{{.*}}-constantargs
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .hword  14
; SmallConstant
; CHECK-NEXT:   .byte   4
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word   -1
; SmallConstant
; CHECK-NEXT:   .byte   4
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word   -1
; SmallConstant
; CHECK-NEXT:   .byte   4
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word   65536
; SmallConstant
; CHECK-NEXT:   .byte   4
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word   2000000000
; SmallConstant
; CHECK-NEXT:   .byte   4
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word   2147483647
; SmallConstant
; CHECK-NEXT:   .byte   4
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word   -1
; SmallConstant
; CHECK-NEXT:   .byte   4
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word   -1
; SmallConstant
; CHECK-NEXT:   .byte   4
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word   0
; LargeConstant at index 0
; CHECK-NEXT:   .byte   5
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word   0
; LargeConstant at index 1
; CHECK-NEXT:   .byte   5
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word   1
; LargeConstant at index 2
; CHECK-NEXT:   .byte   5
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word   2
; SmallConstant
; CHECK-NEXT:   .byte   4
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word   -1
; SmallConstant
; CHECK-NEXT:   .byte   4
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word   66
; LargeConstant at index 2
; CHECK-NEXT:   .byte   5
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word   3

define void @constantargs() {
entry:
  %0 = inttoptr i64 12345 to i8*
  tail call void (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.void(i64 1, i32 16, i8* %0, i32 0, i16 65535, i16 -1, i32 65536, i32 2000000000, i32 2147483647, i32 -1, i32 4294967295, i32 4294967296, i64 2147483648, i64 4294967295, i64 4294967296, i64 -1, i128 66, i128 4294967297)
  ret void
}

; Inline OSR Exit
;
; CHECK-LABEL:  .word   .L{{.*}}-osrinline
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .hword  2
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  {{[0-9]+}}
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word   0
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  {{[0-9]+}}
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word  0
define void @osrinline(i64 %a, i64 %b) {
entry:
  ; Runtime void->void call.
  call void inttoptr (i64 -559038737 to void ()*)()
  ; Followed by inline OSR patchpoint with 12-byte shadow and 2 live vars.
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 3, i32 12, i64 %a, i64 %b)
  ret void
}

; Cold OSR Exit
;
; 2 live variables in register.
;
; CHECK-LABEL:  .word   .L{{.*}}-osrcold
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .hword  2
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  {{[0-9]+}}
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word   0
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  {{[0-9]+}}
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word   0
define void @osrcold(i64 %a, i64 %b) {
entry:
  %test = icmp slt i64 %a, %b
  br i1 %test, label %ret, label %cold
cold:
  ; OSR patchpoint with 12-byte nop-slide and 2 live vars.
  %thunk = inttoptr i64 3735928559 to i8*
  call void (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.void(i64 4, i32 16, i8* %thunk, i32 0, i64 %a, i64 %b)
  unreachable
ret:
  ret void
}

; Property Read
; CHECK-LABEL:  .word   .L{{.*}}-propertyRead
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .hword  2
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  {{[0-9]+}}
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word   0
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  {{[0-9]+}}
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word   0
define i64 @propertyRead(i64* %obj) {
entry:
  %resolveRead = inttoptr i64 3735928559 to i8*
  %result = call anyregcc i64 (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.i64(i64 5, i32 16, i8* %resolveRead, i32 1, i64* %obj)
  %add = add i64 %result, 3
  ret i64 %add
}

; Property Write
; CHECK-LABEL:  .word   .L{{.*}}-propertyWrite
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .hword  2
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  {{[0-9]+}}
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word   0
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  {{[0-9]+}}
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word   0
define void @propertyWrite(i64 %dummy1, i64* %obj, i64 %dummy2, i64 %a) {
entry:
  %resolveWrite = inttoptr i64 3735928559 to i8*
  call anyregcc void (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.void(i64 6, i32 16, i8* %resolveWrite, i32 2, i64* %obj, i64 %a)
  ret void
}

; Void JS Call
;
; 2 live variables in registers.
;
; CHECK-LABEL:  .word   .L{{.*}}-jsVoidCall
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .hword  2
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  {{[0-9]+}}
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word   0
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  {{[0-9]+}}
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word   0
define void @jsVoidCall(i64 %dummy1, i64* %obj, i64 %arg, i64 %l1, i64 %l2) {
entry:
  %resolveCall = inttoptr i64 3735928559 to i8*
  call void (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.void(i64 7, i32 16, i8* %resolveCall, i32 2, i64* %obj, i64 %arg, i64 %l1, i64 %l2)
  ret void
}

; i64 JS Call
;
; 2 live variables in registers.
;
; CHECK-LABEL:  .word   .L{{.*}}-jsIntCall
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .hword  2
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  {{[0-9]+}}
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word   0
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  {{[0-9]+}}
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word   0
define i64 @jsIntCall(i64 %dummy1, i64* %obj, i64 %arg, i64 %l1, i64 %l2) {
entry:
  %resolveCall = inttoptr i64 3735928559 to i8*
  %result = call i64 (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.i64(i64 8, i32 16, i8* %resolveCall, i32 2, i64* %obj, i64 %arg, i64 %l1, i64 %l2)
  %add = add i64 %result, 3
  ret i64 %add
}

; Spilled stack map values.
;
; Verify 28 stack map entries.
;
; CHECK-LABEL:  .word .L{{.*}}-spilledValue
; CHECK-NEXT:   .hword 0
; CHECK-NEXT:   .hword 28
;
; Check that at least one is a spilled entry from RBP.
; Location: Indirect RBP + ...
; CHECK:        .byte 3
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .hword 8
; CHECK-NEXT:   .hword 29
; CHECK-NEXT:   .hword 0
; CHECK-NEXT:   .word
define void @spilledValue(i64 %arg0, i64 %arg1, i64 %arg2, i64 %arg3, i64 %arg4, i64 %l0, i64 %l1, i64 %l2, i64 %l3, i64 %l4, i64 %l5, i64 %l6, i64 %l7, i64 %l8, i64 %l9, i64 %l10, i64 %l11, i64 %l12, i64 %l13, i64 %l14, i64 %l15, i64 %l16, i64 %l17, i64 %l18, i64 %l19, i64 %l20, i64 %l21, i64 %l22, i64 %l23, i64 %l24, i64 %l25, i64 %l26, i64 %l27) {
entry:
  call void (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.void(i64 11, i32 20, i8* null, i32 5, i64 %arg0, i64 %arg1, i64 %arg2, i64 %arg3, i64 %arg4, i64 %l0, i64 %l1, i64 %l2, i64 %l3, i64 %l4, i64 %l5, i64 %l6, i64 %l7, i64 %l8, i64 %l9, i64 %l10, i64 %l11, i64 %l12, i64 %l13, i64 %l14, i64 %l15, i64 %l16, i64 %l17, i64 %l18, i64 %l19, i64 %l20, i64 %l21, i64 %l22, i64 %l23, i64 %l24, i64 %l25, i64 %l26, i64 %l27)
  ret void
}

; Spilled stack map values.
;
; Verify 30 stack map entries.
;
; CHECK-LABEL:  .word .L{{.*}}-spilledStackMapValue
; CHECK-NEXT:   .hword 0
; CHECK-NEXT:   .hword 30
;
; Check that at least one is a spilled entry from RBP.
; Location: Indirect RBP + ...
; CHECK:        .byte 3
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword 8
; CHECK-NEXT:   .hword 29
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word
define webkit_jscc void @spilledStackMapValue(i64 %l0, i64 %l1, i64 %l2, i64 %l3, i64 %l4, i64 %l5, i64 %l6, i64 %l7, i64 %l8, i64 %l9, i64 %l10, i64 %l11, i64 %l12, i64 %l13, i64 %l14, i64 %l15, i64 %l16, i64 %l17, i64 %l18, i64 %l19, i64 %l20, i64 %l21, i64 %l22, i64 %l23, i64 %l24, i64 %l25, i64 %l26, i64 %l27, i64 %l28, i64 %l29) {
entry:
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 12, i32 16, i64 %l0, i64 %l1, i64 %l2, i64 %l3, i64 %l4, i64 %l5, i64 %l6, i64 %l7, i64 %l8, i64 %l9, i64 %l10, i64 %l11, i64 %l12, i64 %l13, i64 %l14, i64 %l15, i64 %l16, i64 %l17, i64 %l18, i64 %l19, i64 %l20, i64 %l21, i64 %l22, i64 %l23, i64 %l24, i64 %l25, i64 %l26, i64 %l27, i64 %l28, i64 %l29)
  ret void
}

; Map a constant value.
;
; CHECK-LABEL:  .word .L{{.*}}-liveConstant
; CHECK-NEXT:   .hword 0
; 1 location
; CHECK-NEXT:   .hword 1
; Loc 0: SmallConstant
; CHECK-NEXT:   .byte   4
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word   33

define void @liveConstant() {
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 15, i32 4, i32 33)
  ret void
}

; Directly map an alloca's address.
;
; Callsite 16
; CHECK-LABEL:  .word .L{{.*}}-directFrameIdx
; CHECK-NEXT:   .hword 0
; 1 location
; CHECK-NEXT:   .hword  1
; Loc 0: Direct RBP - ofs
; CHECK-NEXT:   .byte   2
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  29
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word

; Callsite 17
; CHECK-LABEL:  .word   .L{{.*}}-directFrameIdx
; CHECK-NEXT:   .hword  0
; 2 locations
; CHECK-NEXT:   .hword  2
; Loc 0: Direct RBP - ofs
; CHECK-NEXT:   .byte   2
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  29
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word
; Loc 1: Direct RBP - ofs
; CHECK-NEXT:   .byte   2
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  8
; CHECK-NEXT:   .hword  29
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word
define void @directFrameIdx() {
entry:
  %metadata1 = alloca i64, i32 3, align 8
  store i64 11, i64* %metadata1
  store i64 12, i64* %metadata1
  store i64 13, i64* %metadata1
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 16, i32 0, i64* %metadata1)
  %metadata2 = alloca i8, i32 4, align 8
  %metadata3 = alloca i16, i32 4, align 8
  call void (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.void(i64 17, i32 4, i8* null, i32 0, i8* %metadata2, i16* %metadata3)
  ret void
}

; Test a 64-bit ID.
;
; CHECK:        .xword 4294967295
; CHECK-LABEL:  .word .L{{.*}}-longid
; CHECK:        .xword 4294967296
; CHECK-LABEL:  .word .L{{.*}}-longid
; CHECK:        .xword 9223372036854775807
; CHECK-LABEL:  .word .L{{.*}}-longid
; CHECK:        .xword -1
; CHECK-LABEL:  .word .L{{.*}}-longid
define void @longid() {
entry:
  tail call void (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.void(i64 4294967295, i32 0, i8* null, i32 0)
  tail call void (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.void(i64 4294967296, i32 0, i8* null, i32 0)
  tail call void (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.void(i64 9223372036854775807, i32 0, i8* null, i32 0)
  tail call void (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.void(i64 -1, i32 0, i8* null, i32 0)
  ret void
}

; Map a value when R11 is the only free register.
; The scratch register should not be used for a live stackmap value.
;
; CHECK-LABEL:  .word .L{{.*}}-clobberLR
; CHECK-NEXT:   .hword 0
; 1 location
; CHECK-NEXT:   .hword 1
; Loc 0: Indirect fp - offset
; CHECK-NEXT:   .byte   3
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .hword  4
; CHECK-NEXT:   .hword  29
; CHECK-NEXT:   .hword  0
; CHECK-NEXT:   .word   -{{[0-9]+}}
define void @clobberLR(i32 %a) {
  tail call void asm sideeffect "nop", "~{x0},~{x1},~{x2},~{x3},~{x4},~{x5},~{x6},~{x7},~{x8},~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x15},~{x16},~{x17},~{x18},~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28},~{x29},~{x31}"() nounwind
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 16, i32 8, i32 %a)
  ret void
}

; A stack frame which needs to be realigned at runtime (to meet alignment 
; criteria for values on the stack) does not have a fixed frame size. 
; CHECK-LABEL:  .word .L{{.*}}-needsStackRealignment
; CHECK-NEXT:   .hword 0
; 0 locations
; CHECK-NEXT:   .hword 0
define void @needsStackRealignment() {
  %val = alloca i64, i32 3, align 128
  tail call void (...) @escape_values(i64* %val)
; Note: Adding any non-constant to the stackmap would fail because we
; expected to be able to address off the frame pointer.  In a realigned
; frame, we must use the stack pointer instead.  This is a separate bug.
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 0, i32 0)
  ret void
}
declare void @escape_values(...)

declare void @llvm.experimental.stackmap(i64, i32, ...)
declare void @llvm.experimental.patchpoint.void(i64, i32, i8*, i32, ...)
declare i64 @llvm.experimental.patchpoint.i64(i64, i32, i8*, i32, ...)
