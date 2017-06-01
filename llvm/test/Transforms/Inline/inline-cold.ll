; RUN: opt < %s -inline -S -inlinecold-threshold=75 | FileCheck %s
; Test that functions with attribute Cold are not inlined while the 
; same function without attribute Cold will be inlined.

; RUN: opt < %s -inline -S -inline-threshold=600 | FileCheck %s -check-prefix=OVERRIDE
; The command line argument for inline-threshold should override
; the default cold threshold, so a cold function with size bigger
; than the default cold threshold (225) will be inlined.

; RUN: opt < %s -inline -S | FileCheck %s -check-prefix=DEFAULT
; The same cold function will not be inlined with the default behavior.

@a = global i32 4

; This function should be larger than the cold threshold (75), but smaller
; than the regular threshold.
; Function Attrs: nounwind readnone uwtable
define i32 @simpleFunction(i32 %a) #0 {
entry:
  call void @extern()
  %a1 = load volatile i32, i32* @a
  %x1 = add i32 %a1,  %a1
  %a2 = load volatile i32, i32* @a
  %x2 = add i32 %x1, %a2
  %a3 = load volatile i32, i32* @a
  %x3 = add i32 %x2, %a3
  %a4 = load volatile i32, i32* @a
  %x4 = add i32 %x3, %a4
  %a5 = load volatile i32, i32* @a
  %x5 = add i32 %x4, %a5
  %a6 = load volatile i32, i32* @a
  %x6 = add i32 %x5, %a6
  %a7 = load volatile i32, i32* @a
  %x7 = add i32 %x6, %a6
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
  %add = add i32 %x12, %a
  ret i32 %add
}

; Function Attrs: nounwind cold readnone uwtable
define i32 @ColdFunction(i32 %a) #1 {
; CHECK-LABEL: @ColdFunction
; CHECK: ret
; OVERRIDE-LABEL: @ColdFunction
; OVERRIDE: ret
; DEFAULT-LABEL: @ColdFunction
; DEFAULT: ret
entry:
  call void @extern()
  %a1 = load volatile i32, i32* @a
  %x1 = add i32 %a1,  %a1
  %a2 = load volatile i32, i32* @a
  %x2 = add i32 %x1, %a2
  %a3 = load volatile i32, i32* @a
  %x3 = add i32 %x2, %a3
  %a4 = load volatile i32, i32* @a
  %x4 = add i32 %x3, %a4
  %a5 = load volatile i32, i32* @a
  %x5 = add i32 %x4, %a5
  %a6 = load volatile i32, i32* @a
  %x6 = add i32 %x5, %a6
  %a7 = load volatile i32, i32* @a
  %x7 = add i32 %x6, %a6
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
  %add = add i32 %x12, %a
  ret i32 %add
}

; This function should be larger than the default cold threshold (225).
define i32 @ColdFunction2(i32 %a) #1 {
; CHECK-LABEL: @ColdFunction2
; CHECK: ret
; OVERRIDE-LABEL: @ColdFunction2
; OVERRIDE: ret
; DEFAULT-LABEL: @ColdFunction2
; DEFAULT: ret
entry:
  call void @extern()
  %a1 = load volatile i32, i32* @a
  %x1 = add i32 %a1,  %a1
  %a2 = load volatile i32, i32* @a
  %x2 = add i32 %x1, %a2
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

  %a21 = load volatile i32, i32* @a
  %x21 = add i32 %x12, %a21
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
  %a30 = load volatile i32, i32* @a
  %x30 = add i32 %x29, %a30
  %a31 = load volatile i32, i32* @a
  %x31 = add i32 %x30, %a31
  %a32 = load volatile i32, i32* @a
  %x32 = add i32 %x31, %a32

  %a41 = load volatile i32, i32* @a
  %x41 = add i32 %x32, %a41
  %a42 = load volatile i32, i32* @a
  %x42 = add i32 %x41, %a42
  %a43 = load volatile i32, i32* @a
  %x43 = add i32 %x42, %a43
  %a44 = load volatile i32, i32* @a
  %x44 = add i32 %x43, %a44
  %a45 = load volatile i32, i32* @a
  %x45 = add i32 %x44, %a45
  %a46 = load volatile i32, i32* @a
  %x46 = add i32 %x45, %a46
  %a47 = load volatile i32, i32* @a
  %x47 = add i32 %x46, %a47
  %a48 = load volatile i32, i32* @a
  %x48 = add i32 %x47, %a48
  %a49 = load volatile i32, i32* @a
  %x49 = add i32 %x48, %a49
  %a50 = load volatile i32, i32* @a
  %x50 = add i32 %x49, %a50
  %a51 = load volatile i32, i32* @a
  %x51 = add i32 %x50, %a51
  %a52 = load volatile i32, i32* @a
  %x52 = add i32 %x51, %a52

  %add = add i32 %x52, %a
  ret i32 %add
}

; Function Attrs: nounwind readnone uwtable
define i32 @bar(i32 %a) #0 {
; CHECK-LABEL: @bar
; CHECK: call i32 @ColdFunction(i32 5)
; CHECK-NOT: call i32 @simpleFunction(i32 6)
; CHECK: call i32 @ColdFunction2(i32 5)
; CHECK: ret
; OVERRIDE-LABEL: @bar
; OVERRIDE-NOT: call i32 @ColdFunction(i32 5)
; OVERRIDE-NOT: call i32 @simpleFunction(i32 6)
; OVERRIDE-NOT: call i32 @ColdFunction2(i32 5)
; OVERRIDE: ret
; DEFAULT-LABEL: @bar
; DEFAULT-NOT: call i32 @ColdFunction(i32 5)
; DEFAULT-NOT: call i32 @simpleFunction(i32 6)
; DEFAULT: call i32 @ColdFunction2(i32 5)
; DEFAULT: ret
entry:
  %0 = tail call i32 @ColdFunction(i32 5)
  %1 = tail call i32 @simpleFunction(i32 6)
  %2 = tail call i32 @ColdFunction2(i32 5)
  %3 = add i32 %0, %1
  %add = add i32 %2, %3
  ret i32 %add
}

declare void @extern()
attributes #0 = { nounwind readnone uwtable }
attributes #1 = { nounwind cold readnone uwtable }
