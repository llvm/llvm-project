; RUN: llc < %s -mtriple=thumbv7-apple-ios -mattr=+thumb2 | FileCheck %s -check-prefix=ALL -check-prefix=CHECK
; RUN: llc < %s -mtriple=thumbv7-apple-ios -mattr=+thumb2 -arm-assume-misaligned-load-store | FileCheck %s -check-prefix=ALL -check-prefix=CONSERVATIVE

@X = external global [0 x i32]          ; <[0 x i32]*> [#uses=5]

define i32 @t1() "frame-pointer"="all" {
; ALL-LABEL: t1:
; ALL: push {r7, lr}
; CHECK: ldrd
; CONSERVATIVE-NOT: ldrd
; CONSERVATIVE-NOT: ldm
; ALL: pop {r7, pc}
        %tmp = load i32, i32* getelementptr ([0 x i32], [0 x i32]* @X, i32 0, i32 0)            ; <i32> [#uses=1]
        %tmp3 = load i32, i32* getelementptr ([0 x i32], [0 x i32]* @X, i32 0, i32 1)           ; <i32> [#uses=1]
        %tmp4 = call i32 @f1( i32 %tmp, i32 %tmp3 )                ; <i32> [#uses=1]
        ret i32 %tmp4
}

define i32 @t2() "frame-pointer"="all" {
; ALL-LABEL: t2:
; ALL: push {r7, lr}
; CHECK: ldm
; CONSERVATIVE-NOT: ldrd
; CONSERVATIVE-NOT: ldm
; ALL: pop {r7, pc}
        %tmp = load i32, i32* getelementptr ([0 x i32], [0 x i32]* @X, i32 0, i32 2)            ; <i32> [#uses=1]
        %tmp3 = load i32, i32* getelementptr ([0 x i32], [0 x i32]* @X, i32 0, i32 3)           ; <i32> [#uses=1]
        %tmp5 = load i32, i32* getelementptr ([0 x i32], [0 x i32]* @X, i32 0, i32 4)           ; <i32> [#uses=1]
        %tmp6 = call i32 @f2( i32 %tmp, i32 %tmp3, i32 %tmp5 )             ; <i32> [#uses=1]
        ret i32 %tmp6
}

define i32 @t3() "frame-pointer"="all" {
; ALL-LABEL: t3:
; ALL: push {r7, lr}
; CHECK: ldm
; CONSERVATIVE-NOT: ldrd
; CONSERVATIVE-NOT: ldm
; ALL: pop {r7, pc}
        %tmp = load i32, i32* getelementptr ([0 x i32], [0 x i32]* @X, i32 0, i32 1)            ; <i32> [#uses=1]
        %tmp3 = load i32, i32* getelementptr ([0 x i32], [0 x i32]* @X, i32 0, i32 2)           ; <i32> [#uses=1]
        %tmp5 = load i32, i32* getelementptr ([0 x i32], [0 x i32]* @X, i32 0, i32 3)           ; <i32> [#uses=1]
        %tmp6 = call i32 @f2( i32 %tmp, i32 %tmp3, i32 %tmp5 )             ; <i32> [#uses=1]
        ret i32 %tmp6
}

@g = common global i32* null

define void @t4(i32 %a0, i32 %a1, i32 %a2) "frame-pointer"="all" {
; ALL-LABEL: t4:
; ALL: stm.w sp, {r0, r1, r2}
; ALL: bl _ext
; ALL: ldm.w sp, {r0, r1, r2}
; ALL: bl _f2
  %arr = alloca [4 x i32], align 4
  %p0 = getelementptr inbounds [4 x i32], [4 x i32]* %arr, i64 0, i64 0
  %p1 = getelementptr inbounds [4 x i32], [4 x i32]* %arr, i64 0, i64 1
  %p2 = getelementptr inbounds [4 x i32], [4 x i32]* %arr, i64 0, i64 2
  store i32* %p0, i32** @g, align 8

  store i32 %a0, i32* %p0, align 4
  store i32 %a1, i32* %p1, align 4
  store i32 %a2, i32* %p2, align 4
  call void @ext()

  %v0 = load i32, i32* %p0, align 4
  %v1 = load i32, i32* %p1, align 4
  %v2 = load i32, i32* %p2, align 4
  call i32 @f2(i32 %v0, i32 %v1, i32 %v2)
  ret void
}

declare i32 @f1(i32, i32)

declare i32 @f2(i32, i32, i32)

declare void @ext()
