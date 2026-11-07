; RUN: llc < %s -mtriple=arm64-apple-darwin -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=arm64-apple-darwin -verify-machineinstrs -O0 -global-isel | FileCheck %s --check-prefix=O0
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-darwin"

declare i32 @use_ptr(i32*) #1

; Expect a probe here due to static object size > 4096.
; Function Attrs: noinline nounwind optnone uwtable
define void @static_test1_probe() #0 {
; CHECK-LABEL: static_test1_probe
; CHECK:  mov	w9, #16384
; CHECK:  adrp    x16, ___chkstk_darwin@GOTPAGE
; CHECK:  ldr     x16, [x16, ___chkstk_darwin@GOTPAGEOFF]
; CHECK:  blr x16
; CHECK-NEXT:  sub sp, sp, #4, lsl #12
; O0:  mov	w9, #16384
; O0:  adrp    x16, ___chkstk_darwin@GOTPAGE
; O0:  ldr     x16, [x16, ___chkstk_darwin@GOTPAGEOFF]
; O0:  blr x16
; O0-NEXT:  sub sp, sp, #4, lsl #12
  %1 = alloca [4096 x i32], align 4
  %2 = getelementptr inbounds [4096 x i32], [4096 x i32]* %1, i32 0, i32 0
  %3 = call i32 @use_ptr(i32* %2)
  ret void
}

; No probe without the function attribute present..
; Function Attrs: noinline nounwind optnone uwtable
define void @static_test1_no_attr() {
; CHECK-LABEL: static_test1_no_attr
; CHECK-NOT: bl ___chkstk_darwin
; O0-NOT: bl ___chkstk_darwin
  %1 = alloca [4096 x i32], align 4
  %2 = getelementptr inbounds [4096 x i32], [4096 x i32]* %1, i32 0, i32 0
  %3 = call i32 @use_ptr(i32* %2)
  ret void
}

; Stack size should be less than 4k, no probe.
; Function Attrs: noinline nounwind optnone uwtable
define void @static_test2_small() #0 {
; CHECK-LABEL: static_test2_small
; CHECK-NOT: bl ___chkstk_darwin
; O0-NOT: bl ___chkstk_darwin
  %1 = alloca [64 x i32], align 4
  %2 = getelementptr inbounds [64 x i32], [64 x i32]* %1, i32 0, i32 0
  %3 = call i32 @use_ptr(i32* %2)
  ret void
}

@g = common local_unnamed_addr global i32* null, align 8

; Check that the LR is saved in the prolog for static allocas when it isn't
; otherwise saved as a normal callee-save reg.
; Function Attrs: nounwind optsize ssp uwtable
define void @test_static_oversize_nocsr(i32* nocapture readnone) local_unnamed_addr #0 {
; CHECK-LABEL: test_static_oversize_nocsr
; CHECK:         stp     x28, x27, [sp, #-16]!
; CHECK:    str x30, [sp, #-16]!
; CHECK:  mov w9, #8000
; CHECK:  adrp    x16, ___chkstk_darwin@GOTPAGE
; CHECK:  ldr     x16, [x16, ___chkstk_darwin@GOTPAGEOFF]
; CHECK:  blr x16
; O0:         stp     x28, x27, [sp, #-16]!
; O0:    str x30, [sp, #-16]!
; O0:  mov w9, #8000
; O0:  adrp    x16, ___chkstk_darwin@GOTPAGE
; O0:  ldr     x16, [x16, ___chkstk_darwin@GOTPAGEOFF]
; O0:  blr x16
  %2 = alloca [2000 x i32], align 4
  %3 = bitcast [2000 x i32]* %2 to i32*
  store i32* %3, i32** @g, align 8
  ret void
}

; Test dynamic sized allocas.
; Function Attrs: nounwind optsize ssp uwtable
define void @test_dynamic(i32* nocapture readnone, i64 %num) local_unnamed_addr #0 {
; CHECK-LABEL: test_dynamic
; CHECK:       mov x9, x1
; CHECK:       adrp    x16, ___chkstk_darwin@GOTPAGE
; CHECK:       ldr     x16, [x16, ___chkstk_darwin@GOTPAGEOFF]
; CHECK-NEXT:  blr x16
; CHECK:       add [[AMOUNT:x[0-9]+]], x{{[0-9]+}}, #15
; CHECK-NEXT:  mov x8, sp
; CHECK-NEXT:  and [[AMOUNT]], [[AMOUNT]], #0xfffffffffffffff0
; CHECK-NEXT:  sub x8, x8, x9
; CHECK-NEXT:  mov sp, x8
; O0:       add [[AMOUNT:x[0-9]+]], x{{[0-9]+}}, #15
; O0-NEXT:  and x9, [[AMOUNT]], #0xfffffffffffffff0
; O0:       adrp    x16, ___chkstk_darwin@GOTPAGE
; O0:       ldr     x16, [x16, ___chkstk_darwin@GOTPAGEOFF]
; O0-NEXT:  blr x16
; O0:  mov [[SPCOPY:x[0-9]+]], sp
; O0:  subs [[ALLOC:x[0-9]+]], [[SPCOPY]], x{{[0-9]+}}
; O0-NEXT:  mov sp, [[ALLOC]]
  %2 = alloca i8, i64 %num, align 16
  %3 = bitcast i8* %2 to i32*
  store i32* %3, i32** @g, align 8
  ret void
}


attributes #0 = { "probe-stack"="__chkstk_darwin" }
