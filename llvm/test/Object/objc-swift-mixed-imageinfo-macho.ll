; RUN: llc -mtriple x86_64-apple-ios -filetype asm -o - %s | FileCheck %s
; REQUIRES: x86-registered-target

; It checks whether the backend generates IMAGE_INFO from Swift ABI version + major + minor + "Objective-C Garbage Collection".

target triple = "x86_64-apple-macosx10.15.0"

@llvm.used = appending global [1 x ptr] [ptr @__swift_reflection_version], section "llvm.metadata", align 8
@__swift_reflection_version = linkonce_odr hidden constant i16 3

define i32 @main(i32 %0, ptr %1) #0 {
  %3 = bitcast ptr %1 to ptr
  ret i32 0
}

attributes #0 = { "frame-pointer"="all" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" }

!swift.module.flags = !{!0}
!llvm.linker.options = !{!1, !2, !3}
!llvm.asan.globals = !{!4}
!llvm.module.flags = !{!5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16}
!llvm.ident = !{!17}

!0 = !{!"standard-library", i1 false}
!1 = !{!"-lswiftSwiftOnoneSupport"}
!2 = !{!"-lswiftCore"}
!3 = !{!"-lobjc"}
!4 = !{ptr @llvm.used, null, null, i1 false, i1 true}
!5 = !{i32 2, !"SDK Version", [2 x i32] [i32 10, i32 15]}
!6 = !{i32 1, !"Objective-C Version", i32 2}
!7 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!8 = !{i32 1, !"Objective-C Image Info Section", !"__DATA,__objc_imageinfo,regular,no_dead_strip"}
!9 = !{i32 1, !"Objective-C Garbage Collection", i8 0}
!10 = !{i32 1, !"Objective-C Class Properties", i32 64}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{i32 7, !"PIC Level", i32 2}
!13 = !{i32 1, !"Swift Version", i32 7}
!14 = !{i32 1, !"Swift ABI Version", i32 7}
!15 = !{i32 1, !"Swift Major Version", i8 5}
!16 = !{i32 1, !"Swift Minor Version", i8 1}
!17 = !{!"Apple clang version 11.0.0 (clang-1100.0.33.12)"}

; CHECK: .section	__DATA,__objc_imageinfo,regular,no_dead_strip
; CHECK: L_OBJC_IMAGE_INFO:
; CHECK:   .long 0
; CHECK:   .long 83953472
