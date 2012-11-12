; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s

; CHECK:        .section  __DATA,__objc_imageinfo,regular,no_dead_strip
; CHECK-NEXT: L_OBJC_IMAGE_INFO:
; CHECK-NEXT:   .long  0
; CHECK-NEXT:   .long  2

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = metadata !{i32 1, metadata !"Objective-C Version", i32 2}
!1 = metadata !{i32 1, metadata !"Objective-C Image Info Version", i32 0}
!2 = metadata !{i32 1, metadata !"Objective-C Image Info Section", metadata !"__DATA, __objc_imageinfo, regular, no_dead_strip"}
!3 = metadata !{i32 1, metadata !"Objective-C Garbage Collection", i32 2}
