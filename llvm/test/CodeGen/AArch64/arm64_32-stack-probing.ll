; RUN: llc < %s -mtriple=arm64_32-apple-darwin -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -O0 -mtriple=arm64_32-apple-darwin -verify-machineinstrs | FileCheck %s --check-prefixes=O0

declare i32 @use_ptr(i32*) #1

; Expect a probe here due to static object size > 4096.
; Function Attrs: noinline nounwind optnone uwtable
define void @static_test1_probe() #0 {
; CHECK-LABEL: static_test1_probe
; CHECK:  mov	w9, #16384
; CHECK:  adrp    x16, ___chkstk_darwin@GOTPAGE
; CHECK:  ldr     w16, [x16, ___chkstk_darwin@GOTPAGEOFF]
; CHECK:  blr x16
; CHECK-NEXT:  sub sp, sp, #4, lsl #12
; O0:  mov	w9, #16384
; O0:  adrp    x16, ___chkstk_darwin@GOTPAGE
; O0:  ldr     w16, [x16, ___chkstk_darwin@GOTPAGEOFF]
; O0:  blr x16
; O0-NEXT:  sub sp, sp, #4, lsl #12
  %1 = alloca [4096 x i32], align 4
  %2 = getelementptr inbounds [4096 x i32], [4096 x i32]* %1, i32 0, i32 0
  %3 = call i32 @use_ptr(i32* %2)
  ret void
}


attributes #0 = { "probe-stack"="__chkstk_darwin" nounwind }
