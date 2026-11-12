; RUN: rm -rf %t && split-file %s %t && cd %t

; Clang sets "Objective-C Enforce ClassRO Pointer Signing" to 16 (SignedClassRO)
; when it signs class_ro_t pointers, as it does on arm64e, and to 0 otherwise.
; The value is ORed into the __objc_imageinfo flags, like Class Properties (64).

;--- signed.ll

; RUN: llc < signed.ll | FileCheck %s --check-prefix=SIGNED

; SIGNED:      .section __DATA,__objc_imageinfo,regular,no_dead_strip
; SIGNED-NEXT: L_OBJC_IMAGE_INFO:
; SIGNED-NEXT:   .long 0
; SIGNED-NEXT:   .long 80

target triple = "arm64e-apple-ios"

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5}

!0 = !{i32 1, !"Objective-C Version", i32 2}
!1 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!2 = !{i32 1, !"Objective-C Image Info Section", !"__DATA,__objc_imageinfo,regular,no_dead_strip"}
!3 = !{i32 1, !"Objective-C Garbage Collection", i8 0}
!4 = !{i32 1, !"Objective-C Class Properties", i32 64}
!5 = !{i32 1, !"Objective-C Enforce ClassRO Pointer Signing", i32 16}

;--- unsigned.ll

; RUN: llc < unsigned.ll | FileCheck %s --check-prefix=UNSIGNED

; UNSIGNED:      .section __DATA,__objc_imageinfo,regular,no_dead_strip
; UNSIGNED-NEXT: L_OBJC_IMAGE_INFO:
; UNSIGNED-NEXT:   .long 0
; UNSIGNED-NEXT:   .long 64

target triple = "arm64-apple-ios"

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5}

!0 = !{i32 1, !"Objective-C Version", i32 2}
!1 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!2 = !{i32 1, !"Objective-C Image Info Section", !"__DATA,__objc_imageinfo,regular,no_dead_strip"}
!3 = !{i32 1, !"Objective-C Garbage Collection", i8 0}
!4 = !{i32 1, !"Objective-C Class Properties", i32 64}
!5 = !{i32 1, !"Objective-C Enforce ClassRO Pointer Signing", i8 0}
