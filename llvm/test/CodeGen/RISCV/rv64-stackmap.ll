; RUN: llc -mtriple=riscv64 < %s | FileCheck %s

; CHECK-LABEL:  .section	.llvm_stackmaps
; CHECK-NEXT:  __LLVM_StackMaps:
; Header
; CHECK-NEXT:   .byte   3
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .half   0
; Num Functions
; CHECK-NEXT:   .word   12
; Num LargeConstants
; CHECK-NEXT:   .word   2
; Num Callsites
; CHECK-NEXT:   .word   16

; Functions and stack size
; CHECK-NEXT:   .quad   constantargs
; CHECK-NEXT:   .quad   0
; CHECK-NEXT:   .quad   1
; CHECK-NEXT:   .quad   osrinline
; CHECK-NEXT:   .quad   32
; CHECK-NEXT:   .quad   1
; CHECK-NEXT:   .quad   osrcold
; CHECK-NEXT:   .quad   0
; CHECK-NEXT:   .quad   1
; CHECK-NEXT:   .quad   propertyRead
; CHECK-NEXT:   .quad   16
; CHECK-NEXT:   .quad   1
; CHECK-NEXT:   .quad   propertyWrite
; CHECK-NEXT:   .quad   0
; CHECK-NEXT:   .quad   1
; CHECK-NEXT:   .quad   jsVoidCall
; CHECK-NEXT:   .quad   0
; CHECK-NEXT:   .quad   1
; CHECK-NEXT:   .quad   jsIntCall
; CHECK-NEXT:   .quad   0
; CHECK-NEXT:   .quad   1
; CHECK-NEXT:   .quad   liveConstant
; CHECK-NEXT:   .quad   0
; CHECK-NEXT:   .quad   1
; CHECK-NEXT:   .quad   spilledValue
; CHECK-NEXT:   .quad   144
; CHECK-NEXT:   .quad   1
; CHECK-NEXT:   .quad   directFrameIdx
; CHECK-NEXT:   .quad   48
; CHECK-NEXT:   .quad   2
; CHECK-NEXT:   .quad   longid
; CHECK-NEXT:   .quad   0
; CHECK-NEXT:   .quad   4
; CHECK-NEXT:   .quad   needsStackRealignment
; CHECK-NEXT:   .quad   -1
; CHECK-NEXT:   .quad   1

; Num LargeConstants
; CHECK-NEXT:   .quad   4294967295
; CHECK-NEXT:   .quad   4294967296

; Constant arguments
;
; CHECK-NEXT:   .quad   1
; CHECK-NEXT:   .word   .L{{.*}}-constantargs
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .half   4
; SmallConstant
; CHECK-NEXT:   .byte   4
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .half   8
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .word   65535
; SmallConstant
; CHECK-NEXT:   .byte   4
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .half   8
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .word   65536
; SmallConstant
; CHECK-NEXT:   .byte   5
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .half   8
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .word   0
; LargeConstant at index 0
; CHECK-NEXT:   .byte   5
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .half   8
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .word   1

define void @constantargs() {
entry:
  %0 = inttoptr i64 244837814094590 to ptr
  tail call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 1, i32 28, ptr %0, i32 0, i64 65535, i64 65536, i64 4294967295, i64 4294967296)
  ret void
}

; Inline OSR Exit
;
; CHECK:        .word   .L{{.*}}-osrinline
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .half   2
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .half   8
; CHECK-NEXT:   .half   {{[0-9]+}}
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .word   0
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .half   8
; CHECK-NEXT:   .half   {{[0-9]+}}
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .word   0
define void @osrinline(i64 %a, i64 %b) {
entry:
  ; Runtime void->void call.
  call void inttoptr (i64 244837814094590 to ptr)()
  ; Followed by inline OSR patchpoint with 12-byte shadow and 2 live vars.
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 3, i32 12, i64 %a, i64 %b)
  ret void
}

; Cold OSR Exit
;
; 2 live variables in register.
;
; CHECK:        .word   .L{{.*}}-osrcold
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .half   2
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .half   8
; CHECK-NEXT:   .half   {{[0-9]+}}
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .word   0
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .half   8
; CHECK-NEXT:   .half   {{[0-9]+}}
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .word   0
define void @osrcold(i64 %a, i64 %b) {
entry:
  %test = icmp slt i64 %a, %b
  br i1 %test, label %ret, label %cold
cold:
  ; OSR patchpoint with 28-byte nop-slide and 2 live vars.
  %thunk = inttoptr i64 244837814094590 to ptr
  call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 4, i32 28, ptr %thunk, i32 0, i64 %a, i64 %b)
  unreachable
ret:
  ret void
}

; Property Read
; CHECK-LABEL:  .word   .L{{.*}}-propertyRead
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .half   2
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .half   8
; CHECK-NEXT:   .half   {{[0-9]+}}
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .word   0
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .half   8
; CHECK-NEXT:   .half   {{[0-9]+}}
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .word   0
define i64 @propertyRead(ptr %obj) {
entry:
  %resolveRead = inttoptr i64 244837814094590 to ptr
  %result = call anyregcc i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 5, i32 28, ptr %resolveRead, i32 1, ptr %obj)
  %add = add i64 %result, 3
  ret i64 %add
}

; Property Write
; CHECK:        .word   .L{{.*}}-propertyWrite
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .half   2
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .half   8
; CHECK-NEXT:   .half   {{[0-9]+}}
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .word   0
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .half   8
; CHECK-NEXT:   .half   {{[0-9]+}}
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .word   0
define void @propertyWrite(i64 %dummy1, ptr %obj, i64 %dummy2, i64 %a) {
entry:
  %resolveWrite = inttoptr i64 244837814094590 to ptr
  call anyregcc void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 6, i32 28, ptr %resolveWrite, i32 2, ptr %obj, i64 %a)
  ret void
}

; Void JS Call
;
; 2 live variables in registers.
;
; CHECK:        .word   .L{{.*}}-jsVoidCall
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .half   2
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .half   8
; CHECK-NEXT:   .half   {{[0-9]+}}
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .word   0
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .half   8
; CHECK-NEXT:   .half   {{[0-9]+}}
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .word   0
define void @jsVoidCall(i64 %dummy1, ptr %obj, i64 %arg, i64 %l1, i64 %l2) {
entry:
  %resolveCall = inttoptr i64 244837814094590 to ptr
  call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 7, i32 28, ptr %resolveCall, i32 2, ptr %obj, i64 %arg, i64 %l1, i64 %l2)
  ret void
}

; i64 JS Call
;
; 2 live variables in registers.
;
; CHECK:        .word   .L{{.*}}-jsIntCall
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .half   2
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .half   8
; CHECK-NEXT:   .half   {{[0-9]+}}
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .word   0
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .half   8
; CHECK-NEXT:   .half   {{[0-9]+}}
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .word   0
define i64 @jsIntCall(i64 %dummy1, ptr %obj, i64 %arg, i64 %l1, i64 %l2) {
entry:
  %resolveCall = inttoptr i64 244837814094590 to ptr
  %result = call i64 (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.i64(i64 8, i32 28, ptr %resolveCall, i32 2, ptr %obj, i64 %arg, i64 %l1, i64 %l2)
  %add = add i64 %result, 3
  ret i64 %add
}

; Map a constant value.
;
; CHECK:        .word   .L{{.*}}-liveConstant
; CHECK-NEXT:   .half   0
; 1 location
; CHECK-NEXT:   .half   1
; Loc 0: SmallConstant
; CHECK-NEXT:   .byte   4
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .half   8
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .word   33

define void @liveConstant() {
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 15, i32 8, i32 33)
  ret void
}

; Spilled stack map values.
;
; Verify 28 stack map entries.
;
; CHECK-LABEL:  .word   .L{{.*}}-spilledValue
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .half   28
;
; Check that at least one is a spilled entry from RBP.
; Location: Indirect RBP + ...
; CHECK:        .byte   3
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .half   8
; CHECK-NEXT:   .half   2
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .word
define void @spilledValue(i64 %arg0, i64 %arg1, i64 %arg2, i64 %arg3, i64 %arg4, i64 %l0, i64 %l1, i64 %l2, i64 %l3, i64 %l4, i64 %l5, i64 %l6, i64 %l7, i64 %l8, i64 %l9, i64 %l10, i64 %l11, i64 %l12, i64 %l13, i64 %l14, i64 %l15, i64 %l16, i64 %l17, i64 %l18, i64 %l19, i64 %l20, i64 %l21, i64 %l22, i64 %l23, i64 %l24, i64 %l25, i64 %l26, i64 %l27) {
entry:
  call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 11, i32 28, ptr null, i32 5, i64 %arg0, i64 %arg1, i64 %arg2, i64 %arg3, i64 %arg4, i64 %l0, i64 %l1, i64 %l2, i64 %l3, i64 %l4, i64 %l5, i64 %l6, i64 %l7, i64 %l8, i64 %l9, i64 %l10, i64 %l11, i64 %l12, i64 %l13, i64 %l14, i64 %l15, i64 %l16, i64 %l17, i64 %l18, i64 %l19, i64 %l20, i64 %l21, i64 %l22, i64 %l23, i64 %l24, i64 %l25, i64 %l26, i64 %l27)
  ret void
}

; Directly map an alloca's address.
;
; Callsite 16
; CHECK-LABEL:  .word .L{{.*}}-directFrameIdx
; CHECK-NEXT:   .half   0
; 1 location
; CHECK-NEXT:   .half   1
; Loc 0: Direct RBP - ofs
; CHECK-NEXT:   .byte   2
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .half   8
; CHECK-NEXT:   .half   2
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .word

; Callsite 17
; CHECK-LABEL:  .word   .L{{.*}}-directFrameIdx
; CHECK-NEXT:   .half   0
; 2 locations
; CHECK-NEXT:   .half   2
; Loc 0: Direct RBP - ofs
; CHECK-NEXT:   .byte   2
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .half   8
; CHECK-NEXT:   .half   2
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .word
; Loc 1: Direct RBP - ofs
; CHECK-NEXT:   .byte   2
; CHECK-NEXT:   .byte   0
; CHECK-NEXT:   .half   8
; CHECK-NEXT:   .half   2
; CHECK-NEXT:   .half   0
; CHECK-NEXT:   .word
define void @directFrameIdx() {
entry:
  %metadata1 = alloca i64, i32 3, align 8
  store i64 11, ptr %metadata1
  store i64 12, ptr %metadata1
  store i64 13, ptr %metadata1
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 16, i32 0, ptr %metadata1)
  %metadata2 = alloca i8, i32 4, align 8
  %metadata3 = alloca i16, i32 4, align 8
  call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 17, i32 4, ptr null, i32 0, ptr %metadata2, ptr %metadata3)
  ret void
}

; Test a 64-bit ID.
;
; CHECK:        .quad   4294967295
; CHECK-LABEL:  .word   .L{{.*}}-longid
; CHECK:        .quad   4294967296
; CHECK-LABEL:  .word   .L{{.*}}-longid
; CHECK:        .quad   9223372036854775807
; CHECK-LABEL:  .word   .L{{.*}}-longid
; CHECK:        .quad   -1
; CHECK-LABEL:  .word   .L{{.*}}-longid
define void @longid() {
entry:
  tail call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 4294967295, i32 0, ptr null, i32 0)
  tail call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 4294967296, i32 0, ptr null, i32 0)
  tail call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 9223372036854775807, i32 0, ptr null, i32 0)
  tail call void (i64, i32, ptr, i32, ...) @llvm.experimental.patchpoint.void(i64 -1, i32 0, ptr null, i32 0)
  ret void
}

; A stack frame which needs to be realigned at runtime (to meet alignment
; criteria for values on the stack) does not have a fixed frame size.
; CHECK-LABEL:  .word   .L{{.*}}-needsStackRealignment
; CHECK-NEXT:   .half   0
; 0 locations
; CHECK-NEXT:   .half   0
define void @needsStackRealignment() {
  %val = alloca i64, i32 3, align 128
  tail call void (...) @escape_values(ptr %val)
; Note: Adding any non-constant to the stackmap would fail because we
; expected to be able to address off the frame pointer.  In a realigned
; frame, we must use the stack pointer instead.  This is a separate bug.
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 0, i32 0)
  ret void
}
declare void @escape_values(...)

declare void @llvm.experimental.stackmap(i64, i32, ...)
declare void @llvm.experimental.patchpoint.void(i64, i32, ptr, i32, ...)
declare i64 @llvm.experimental.patchpoint.i64(i64, i32, ptr, i32, ...)
