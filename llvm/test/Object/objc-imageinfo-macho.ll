; RUN: llc -mtriple x86_64-apple-ios -filetype asm -o - %s | FileCheck %s
; REQUIRES: x86-registered-target

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"Objective-C Version", i32 2}
!1 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!2 = !{i32 1, !"Objective-C Image Info Section", !"__DATA,__objc_imageinfo,regular,no_dead_strip"}
!3 = !{i32 1, !"Objective-C Garbage Collection", i32 2}

; CHECK: .section __DATA,__objc_imageinfo,regular,no_dead_strip
; CHECK: L_OBJC_IMAGE_INFO:
; CHECK:   .long 0
; CHECK:   .long 2

