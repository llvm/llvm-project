; RUN: llc < %s -mtriple=armv7-apple-ios -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7-apple-ios -verify-machineinstrs | FileCheck %s

define void @func(i32 %argc, i8** %argv) nounwind {
entry:
	%argc.addr = alloca i32		; <i32*> [#uses=1]
	%argv.addr = alloca i8**		; <i8***> [#uses=1]
	%val1 = alloca i32		; <i32*> [#uses=2]
	%val2 = alloca i32		; <i32*> [#uses=15]
	%andt = alloca i32		; <i32*> [#uses=2]
	%ort = alloca i32		; <i32*> [#uses=2]
	%xort = alloca i32		; <i32*> [#uses=2]
	%old = alloca i32		; <i32*> [#uses=18]
	%temp = alloca i32		; <i32*> [#uses=2]
	store i32 %argc, i32* %argc.addr
	store i8** %argv, i8*** %argv.addr
	store i32 0, i32* %val1
	store i32 31, i32* %val2
	store i32 3855, i32* %andt
	store i32 3855, i32* %ort
	store i32 3855, i32* %xort
	store i32 4, i32* %temp
	%tmp = load i32* %temp
  ; CHECK: ldrex
  ; CHECK: add
  ; CHECK: strex
  %0 = atomicrmw add i32* %val1, i32 %tmp monotonic
	store i32 %0, i32* %old
  ; CHECK: ldrex
  ; CHECK: sub
  ; CHECK: strex
  %1 = atomicrmw sub i32* %val2, i32 30 monotonic
	store i32 %1, i32* %old
  ; CHECK: ldrex
  ; CHECK: add
  ; CHECK: strex
  %2 = atomicrmw add i32* %val2, i32 1 monotonic
	store i32 %2, i32* %old
  ; CHECK: ldrex
  ; CHECK: sub
  ; CHECK: strex
  %3 = atomicrmw sub i32* %val2, i32 1 monotonic
	store i32 %3, i32* %old
  ; CHECK: ldrex
  ; CHECK: and
  ; CHECK: strex
  %4 = atomicrmw and i32* %andt, i32 4080 monotonic
	store i32 %4, i32* %old
  ; CHECK: ldrex
  ; CHECK: or
  ; CHECK: strex
  %5 = atomicrmw or i32* %ort, i32 4080 monotonic
	store i32 %5, i32* %old
  ; CHECK: ldrex
  ; CHECK: eor
  ; CHECK: strex
  %6 = atomicrmw xor i32* %xort, i32 4080 monotonic
	store i32 %6, i32* %old
  ; CHECK: ldrex
  ; CHECK: cmp
  ; CHECK: strex
  %7 = atomicrmw min i32* %val2, i32 16 monotonic
	store i32 %7, i32* %old
	%neg = sub i32 0, 1
  ; CHECK: ldrex
  ; CHECK: cmp
  ; CHECK: strex
  %8 = atomicrmw min i32* %val2, i32 %neg monotonic
	store i32 %8, i32* %old
  ; CHECK: ldrex
  ; CHECK: cmp
  ; CHECK: strex
  %9 = atomicrmw max i32* %val2, i32 1 monotonic
	store i32 %9, i32* %old
  ; CHECK: ldrex
  ; CHECK: cmp
  ; CHECK: strex
  %10 = atomicrmw max i32* %val2, i32 0 monotonic
	store i32 %10, i32* %old
  ; CHECK: ldrex
  ; CHECK: cmp
  ; CHECK: strex
  %11 = atomicrmw umin i32* %val2, i32 16 monotonic
	store i32 %11, i32* %old
	%uneg = sub i32 0, 1
  ; CHECK: ldrex
  ; CHECK: cmp
  ; CHECK: strex
  %12 = atomicrmw umin i32* %val2, i32 %uneg monotonic
	store i32 %12, i32* %old
  ; CHECK: ldrex
  ; CHECK: cmp
  ; CHECK: strex
  %13 = atomicrmw umax i32* %val2, i32 1 monotonic
	store i32 %13, i32* %old
  ; CHECK: ldrex
  ; CHECK: cmp
  ; CHECK: strex
  %14 = atomicrmw umax i32* %val2, i32 0 monotonic
	store i32 %14, i32* %old

  ret void
}

define void @func2() nounwind {
entry:
  %val = alloca i16
  %old = alloca i16
  store i16 31, i16* %val
  ; CHECK: ldrex
  ; CHECK: cmp
  ; CHECK: strex
  %0 = atomicrmw umin i16* %val, i16 16 monotonic
  store i16 %0, i16* %old
  %uneg = sub i16 0, 1
  ; CHECK: ldrex
  ; CHECK: cmp
  ; CHECK: strex
  %1 = atomicrmw umin i16* %val, i16 %uneg monotonic
  store i16 %1, i16* %old
  ; CHECK: ldrex
  ; CHECK: cmp
  ; CHECK: strex
  %2 = atomicrmw umax i16* %val, i16 1 monotonic
  store i16 %2, i16* %old
  ; CHECK: ldrex
  ; CHECK: cmp
  ; CHECK: strex
  %3 = atomicrmw umax i16* %val, i16 0 monotonic
  store i16 %3, i16* %old
  ret void
}

define void @func3() nounwind {
entry:
  %val = alloca i8
  %old = alloca i8
  store i8 31, i8* %val
  ; CHECK: ldrex
  ; CHECK: cmp
  ; CHECK: strex
  %0 = atomicrmw umin i8* %val, i8 16 monotonic
  store i8 %0, i8* %old
  ; CHECK: ldrex
  ; CHECK: cmp
  ; CHECK: strex
  %uneg = sub i8 0, 1
  %1 = atomicrmw umin i8* %val, i8 %uneg monotonic
  store i8 %1, i8* %old
  ; CHECK: ldrex
  ; CHECK: cmp
  ; CHECK: strex
  %2 = atomicrmw umax i8* %val, i8 1 monotonic
  store i8 %2, i8* %old
  ; CHECK: ldrex
  ; CHECK: cmp
  ; CHECK: strex
  %3 = atomicrmw umax i8* %val, i8 0 monotonic
  store i8 %3, i8* %old
  ret void
}

; CHECK: func4
; This function should not need to use callee-saved registers.
; rdar://problem/12203728
; CHECK-NOT: r4
define i32 @func4(i32* %p) nounwind optsize ssp {
entry:
  %0 = atomicrmw add i32* %p, i32 1 monotonic
  ret i32 %0
}
