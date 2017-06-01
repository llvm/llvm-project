; RUN: llc                             < %s | FileCheck %s
;
; Note: Print verbose stackmaps using -debug-only=stackmaps.

; We are not getting the correct stack alignment when cross compiling for arm64.
; So specify a datalayout here.
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; CHECK-LABEL: constantargs:
; CHECK: {{^}}.L[[constantargs_BEGIN:.*]]:{{$}}

; CHECK-LABEL: osrinline:
; CHECK: {{^}}.L[[osrinline_BEGIN:.*]]:{{$}}

; CHECK-LABEL: osrcold:
; CHECK: {{^}}.L[[osrcold_BEGIN:.*]]:{{$}}

; CHECK-LABEL: propertyRead:
; CHECK: {{^}}.L[[propertyRead_BEGIN:.*]]:{{$}}

; CHECK-LABEL: propertyWrite:
; CHECK: {{^}}.L[[propertyWrite_BEGIN:.*]]:{{$}}

; CHECK-LABEL: jsVoidCall:
; CHECK: {{^}}.L[[jsVoidCall_BEGIN:.*]]:{{$}}

; CHECK-LABEL: jsIntCall:
; CHECK: {{^}}.L[[jsIntCall_BEGIN:.*]]:{{$}}

; CHECK-LABEL: spilledValue:
; CHECK: {{^}}.L[[spilledValue_BEGIN:.*]]:{{$}}

; CHECK-LABEL: spilledStackMapValue:
; CHECK: {{^}}.L[[spilledStackMapValue_BEGIN:.*]]:{{$}}

; CHECK-LABEL: liveConstant:
; CHECK: {{^}}.L[[liveConstant_BEGIN:.*]]:{{$}}

; CHECK-LABEL: clobberLR:
; CHECK: {{^}}.L[[clobberLR_BEGIN:.*]]:{{$}}


; CHECK-LABEL:  .section  .llvm_stackmaps
; CHECK-NEXT:  __LLVM_StackMaps:
; Header
; CHECK-NEXT:   .byte 2
; CHECK-NEXT:   .byte 0
; CHECK-NEXT:   .short 0
; Num Functions
; CHECK-NEXT:   .long 11
; Num LargeConstants
; CHECK-NEXT:   .long 2
; Num Callsites
; CHECK-NEXT:   .long 11

; Functions and stack size
; CHECK-NEXT:   .quad constantargs
; CHECK-NEXT:   .quad 128
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad osrinline
; CHECK-NEXT:   .quad 144
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad osrcold
; CHECK-NEXT:   .quad 128
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad propertyRead
; CHECK-NEXT:   .quad 128
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad propertyWrite
; CHECK-NEXT:   .quad 128
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad jsVoidCall
; CHECK-NEXT:   .quad 128
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad jsIntCall
; CHECK-NEXT:   .quad 128
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad spilledValue
; CHECK-NEXT:   .quad 304
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad spilledStackMapValue
; CHECK-NEXT:   .quad 224
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad liveConstant
; CHECK-NEXT:   .quad 64
; CHECK-NEXT:   .quad 1
; CHECK-NEXT:   .quad clobberLR
; CHECK-NEXT:   .quad 208
; CHECK-NEXT:   .quad 1

; Num LargeConstants
; CHECK-NEXT:   .quad   4294967295
; CHECK-NEXT:   .quad   4294967296

; Constant arguments
;
; CHECK-NEXT:   .quad   1
; CHECK-NEXT:   .long   .L{{.*}}-.L[[constantargs_BEGIN]]
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .short  4
; SmallConstant
; CHECK-NEXT:   .byte   4
; CHECK-NEXT:   .byte   8
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .long   65535
; SmallConstant
; CHECK-NEXT:   .byte   4
; CHECK-NEXT:   .byte   8
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .long   65536
; SmallConstant
; CHECK-NEXT:   .byte   5
; CHECK-NEXT:   .byte   8
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .long   0
; LargeConstant at index 0
; CHECK-NEXT:   .byte   5
; CHECK-NEXT:   .byte   8
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .long   1

define void @constantargs() {
entry:
  %0 = inttoptr i64 244837814094590 to i8*
  tail call void (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.void(i64 1, i32 40, i8* %0, i32 0, i64 65535, i64 65536, i64 4294967295, i64 4294967296)
  ret void
}

; Inline OSR Exit
;
; CHECK:  .long   .L{{.*}}-.L[[osrinline_BEGIN]]
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .short  2
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   8
; CHECK-NEXT:   .short  {{[0-9]+}}
; CHECK-NEXT:   .long   0
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   8
; CHECK-NEXT:   .short  {{[0-9]+}}
; CHECK-NEXT:   .long  0
define void @osrinline(i64 %a, i64 %b) {
entry:
  ; Runtime void->void call.
  call void inttoptr (i64 244837814094590 to void ()*)()
  ; Followed by inline OSR patchpoint with 12-byte shadow and 2 live vars.
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 3, i32 12, i64 %a, i64 %b)
  ret void
}

; Cold OSR Exit
;
; 2 live variables in register.
;
; CHECK:  .long   .L{{.*}}-.L[[osrcold_BEGIN]]
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .short  2
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   8
; CHECK-NEXT:   .short  {{[0-9]+}}
; CHECK-NEXT:   .long   0
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   8
; CHECK-NEXT:   .short  {{[0-9]+}}
; CHECK-NEXT:   .long  0
define void @osrcold(i64 %a, i64 %b) {
entry:
  %test = icmp slt i64 %a, %b
  br i1 %test, label %ret, label %cold
cold:
  ; OSR patchpoint with 12-byte nop-slide and 2 live vars.
  %thunk = inttoptr i64 244837814094590 to i8*
  call void (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.void(i64 4, i32 40, i8* %thunk, i32 0, i64 %a, i64 %b)
  unreachable
ret:
  ret void
}

; Property Read
; CHECK:  .long   .L{{.*}}-.L[[propertyRead_BEGIN]]
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .short  0
;
; FIXME: There are currently no stackmap entries. After moving to
; AnyRegCC, we will have entries for the object and return value.
define i64 @propertyRead(i64* %obj) {
entry:
  %resolveRead = inttoptr i64 244837814094590 to i8*
  %result = call i64 (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.i64(i64 5, i32 40, i8* %resolveRead, i32 1, i64* %obj)
  %add = add i64 %result, 3
  ret i64 %add
}

; Property Write
; CHECK:  .long   .L{{.*}}-.L[[propertyWrite_BEGIN]]
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .short  2
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   8
; CHECK-NEXT:   .short  {{[0-9]+}}
; CHECK-NEXT:   .long   0
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   8
; CHECK-NEXT:   .short  {{[0-9]+}}
; CHECK-NEXT:   .long   0
define void @propertyWrite(i64 %dummy1, i64* %obj, i64 %dummy2, i64 %a) {
entry:
  %resolveWrite = inttoptr i64 244837814094590 to i8*
  call anyregcc void (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.void(i64 6, i32 40, i8* %resolveWrite, i32 2, i64* %obj, i64 %a)
  ret void
}

; Void JS Call
;
; 2 live variables in registers.
;
; CHECK:  .long   .L{{.*}}-.L[[jsVoidCall_BEGIN]]
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .short  2
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   8
; CHECK-NEXT:   .short  {{[0-9]+}}
; CHECK-NEXT:   .long   0
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   8
; CHECK-NEXT:   .short  {{[0-9]+}}
; CHECK-NEXT:   .long   0
define void @jsVoidCall(i64 %dummy1, i64* %obj, i64 %arg, i64 %l1, i64 %l2) {
entry:
  %resolveCall = inttoptr i64 244837814094590 to i8*
  call void (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.void(i64 7, i32 40, i8* %resolveCall, i32 2, i64* %obj, i64 %arg, i64 %l1, i64 %l2)
  ret void
}

; i64 JS Call
;
; 2 live variables in registers.
;
; CHECK:  .long   .L{{.*}}-.L[[jsIntCall_BEGIN]]
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .short  2
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   8
; CHECK-NEXT:   .short  {{[0-9]+}}
; CHECK-NEXT:   .long   0
; CHECK-NEXT:   .byte   1
; CHECK-NEXT:   .byte   8
; CHECK-NEXT:   .short  {{[0-9]+}}
; CHECK-NEXT:   .long   0
define i64 @jsIntCall(i64 %dummy1, i64* %obj, i64 %arg, i64 %l1, i64 %l2) {
entry:
  %resolveCall = inttoptr i64 244837814094590 to i8*
  %result = call i64 (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.i64(i64 8, i32 40, i8* %resolveCall, i32 2, i64* %obj, i64 %arg, i64 %l1, i64 %l2)
  %add = add i64 %result, 3
  ret i64 %add
}

; Spilled stack map values.
;
; Verify 28 stack map entries.
;
; CHECK:  .long .L{{.*}}-.L[[spilledValue_BEGIN]]
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .short 28
;
; Check that at least one is a spilled entry from r31.
; Location: Indirect FP + ...
; CHECK:        .byte 3
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short 31
define void @spilledValue(i64 %arg0, i64 %arg1, i64 %arg2, i64 %arg3, i64 %arg4, i64 %l0, i64 %l1, i64 %l2, i64 %l3, i64 %l4, i64 %l5, i64 %l6, i64 %l7, i64 %l8, i64 %l9, i64 %l10, i64 %l11, i64 %l12, i64 %l13, i64 %l14, i64 %l15, i64 %l16, i64 %l17, i64 %l18, i64 %l19, i64 %l20, i64 %l21, i64 %l22, i64 %l23, i64 %l24, i64 %l25, i64 %l26, i64 %l27) {
entry:
  call void (i64, i32, i8*, i32, ...) @llvm.experimental.patchpoint.void(i64 11, i32 40, i8* null, i32 5, i64 %arg0, i64 %arg1, i64 %arg2, i64 %arg3, i64 %arg4, i64 %l0, i64 %l1, i64 %l2, i64 %l3, i64 %l4, i64 %l5, i64 %l6, i64 %l7, i64 %l8, i64 %l9, i64 %l10, i64 %l11, i64 %l12, i64 %l13, i64 %l14, i64 %l15, i64 %l16, i64 %l17, i64 %l18, i64 %l19, i64 %l20, i64 %l21, i64 %l22, i64 %l23, i64 %l24, i64 %l25, i64 %l26, i64 %l27)
  ret void
}

; Spilled stack map values.
;
; Verify 30 stack map entries.
;
; CHECK:  .long .L{{.*}}-.L[[spilledStackMapValue_BEGIN]]
; CHECK-NEXT:   .short 0
; CHECK-NEXT:   .short 30
;
; Check that at least one is a spilled entry from r31.
; Location: Indirect FP + ...
; CHECK:        .byte 3
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .short 31
define webkit_jscc void @spilledStackMapValue(i64 %l0, i64 %l1, i64 %l2, i64 %l3, i64 %l4, i64 %l5, i64 %l6, i64 %l7, i64 %l8, i64 %l9, i64 %l10, i64 %l11, i64 %l12, i64 %l13, i64 %l14, i64 %l15, i64 %l16, i64 %l17, i64 %l18, i64 %l19, i64 %l20, i64 %l21, i64 %l22, i64 %l23, i64 %l24, i64 %l25, i64 %l26, i64 %l27, i64 %l28, i64 %l29) {
entry:
  call void (i64, i32, ...) @llvm.experimental.stackmap(i64 12, i32 16, i64 %l0, i64 %l1, i64 %l2, i64 %l3, i64 %l4, i64 %l5, i64 %l6, i64 %l7, i64 %l8, i64 %l9, i64 %l10, i64 %l11, i64 %l12, i64 %l13, i64 %l14, i64 %l15, i64 %l16, i64 %l17, i64 %l18, i64 %l19, i64 %l20, i64 %l21, i64 %l22, i64 %l23, i64 %l24, i64 %l25, i64 %l26, i64 %l27, i64 %l28, i64 %l29)
  ret void
}


; Map a constant value.
;
; CHECK:  .long .L{{.*}}-.L[[liveConstant_BEGIN]]
; CHECK-NEXT:   .short 0
; 1 location
; CHECK-NEXT:   .short 1
; Loc 0: SmallConstant
; CHECK-NEXT:   .byte   4
; CHECK-NEXT:   .byte   8
; CHECK-NEXT:   .short  0
; CHECK-NEXT:   .long   33

define void @liveConstant() {
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 15, i32 8, i32 33)
  ret void
}

; Map a value when LR is the only free register.
;
; CHECK:  .long .L{{.*}}-.L[[clobberLR_BEGIN]]
; CHECK-NEXT:   .short 0
; 1 location
; CHECK-NEXT:   .short 1
; Loc 0: Indirect FP (r31) - offset
; CHECK-NEXT:   .byte   3
; CHECK-NEXT:   .byte   4
; CHECK-NEXT:   .short  31
; CHECK-NEXT:   .long   {{[0-9]+}}
define void @clobberLR(i32 %a) {
  tail call void asm sideeffect "nop", "~{r0},~{r3},~{r4},~{r5},~{r6},~{r7},~{r8},~{r9},~{r10},~{r11},~{r12},~{r14},~{r15},~{r16},~{r17},~{r18},~{r19},~{r20},~{r21},~{r22},~{r23},~{r24},~{r25},~{r26},~{r27},~{r28},~{r29},~{r30},~{r31}"() nounwind
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 16, i32 8, i32 %a)
  ret void
}

declare void @llvm.experimental.stackmap(i64, i32, ...)
declare void @llvm.experimental.patchpoint.void(i64, i32, i8*, i32, ...)
declare i64 @llvm.experimental.patchpoint.i64(i64, i32, i8*, i32, ...)
