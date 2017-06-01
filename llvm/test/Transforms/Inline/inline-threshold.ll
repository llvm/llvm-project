; Test that -inline-threshold overrides thresholds derived from opt levels.
; RUN: opt < %s -O2 -inline-threshold=500 -S  | FileCheck %s
; RUN: opt < %s -O3 -inline-threshold=500 -S  | FileCheck %s
; RUN: opt < %s -Os -inline-threshold=500 -S  | FileCheck %s
; RUN: opt < %s -Oz -inline-threshold=500 -S  | FileCheck %s

@a = global i32 4

define i32 @simpleFunction(i32 %a) #0 {
entry:
  %a1 = load volatile i32, i32* @a
  %x1 = add i32 %a1,  %a1
  %cmp = icmp eq i32 %a1, 0
  br i1 %cmp, label %if.then, label %if.else
if.then:
  %a2 = load volatile i32, i32* @a
  %x2_0 = add i32 %x1, %a2
  br label %if.else
if.else:
  %x2 = phi i32 [ %x1, %entry ], [ %x2_0, %if.then ]
  %a3 = load volatile i32, i32* @a
  %x3 = add i32 %x2, %a3
  %a4 = load volatile i32, i32* @a
  %x4 = add i32 %x3, %a4
  %a5 = load volatile i32, i32* @a
  %x5 = add i32 %x4, %a5
  %a6 = load volatile i32, i32* @a
  %x6 = add i32 %x5, %a6
  %a7 = load volatile i32, i32* @a
  %x7 = add i32 %x6, %a7
  %a8 = load volatile i32, i32* @a
  %x8 = add i32 %x7, %a8
  %a9 = load volatile i32, i32* @a
  %x9 = add i32 %x8, %a9
  %a10 = load volatile i32, i32* @a
  %x10 = add i32 %x9, %a10
  %a11 = load volatile i32, i32* @a
  %x11 = add i32 %x10, %a11
  %a12 = load volatile i32, i32* @a
  %x12 = add i32 %x11, %a12
  %a13 = load volatile i32, i32* @a
  %x13 = add i32 %x12, %a13
  %a14 = load volatile i32, i32* @a
  %x14 = add i32 %x13, %a14
  %a15 = load volatile i32, i32* @a
  %x15 = add i32 %x14, %a15
  %a16 = load volatile i32, i32* @a
  %x16 = add i32 %x15, %a16
  %a17 = load volatile i32, i32* @a
  %x17 = add i32 %x16, %a17
  %a18 = load volatile i32, i32* @a
  %x18 = add i32 %x17, %a18
  %a19 = load volatile i32, i32* @a
  %x19 = add i32 %x18, %a19
  %a20 = load volatile i32, i32* @a
  %x20 = add i32 %x19, %a20
  %a21 = load volatile i32, i32* @a
  %x21 = add i32 %x20, %a21
  %a22 = load volatile i32, i32* @a
  %x22 = add i32 %x21, %a22
  %a23 = load volatile i32, i32* @a
  %x23 = add i32 %x22, %a23
  %a24 = load volatile i32, i32* @a
  %x24 = add i32 %x23, %a24
  %a25 = load volatile i32, i32* @a
  %x25 = add i32 %x24, %a25
  %a26 = load volatile i32, i32* @a
  %x26 = add i32 %x25, %a26
  %a27 = load volatile i32, i32* @a
  %x27 = add i32 %x26, %a27
  %a28 = load volatile i32, i32* @a
  %x28 = add i32 %x27, %a28
  %a29 = load volatile i32, i32* @a
  %x29 = add i32 %x28, %a29
  %add = add i32 %x29, %a
  ret i32 %add
}

; Function Attrs: nounwind readnone uwtable
define i32 @bar(i32 %a) #0 {
; CHECK-LABEL: @bar
; CHECK-NOT: call i32 @simpleFunction(i32 6)
; CHECK: ret
entry:
  %i = tail call i32 @simpleFunction(i32 6)
  ret i32 %i
}

attributes #0 = { nounwind readnone uwtable }
