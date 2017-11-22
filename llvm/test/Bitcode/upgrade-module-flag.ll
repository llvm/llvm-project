; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder < %s

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"PIC Level", i32 1}
!1 = !{i32 1, !"PIE Level", i32 1}
!2 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!3 = !{i32 1, !"Objective-C Image Info Section", !"__DATA, __objc_imageinfo, regular, no_dead_strip"}

; CHECK: !0 = !{i32 7, !"PIC Level", i32 1}
; CHECK: !1 = !{i32 7, !"PIE Level", i32 1}
; CHECK: !2 = !{i32 1, !"Objective-C Image Info Version", i32 0}
; CHECK: !3 = !{i32 1, !"Objective-C Image Info Section", !"__DATA,__objc_imageinfo,regular,no_dead_strip"}
; CHECK: !4 = !{i32 4, !"Objective-C Class Properties", i32 0}
