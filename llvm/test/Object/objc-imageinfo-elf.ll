; RUN: llc -mtriple x86_64-unknown-linux-gnu -filetype asm -o - %s | FileCheck %s
; REQUIRES: x86-registered-target

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"Objective-C Version", i32 2}
!1 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!2 = !{i32 1, !"Objective-C Image Info Section", !"objc_imageinfo"}
!3 = !{i32 1, !"Objective-C Garbage Collection", i32 2}

; CHECK: .section objc_imageinfo
; CHECK: OBJC_IMAGE_INFO:
; CHECK:   .long 0
; CHECK:   .long 2

