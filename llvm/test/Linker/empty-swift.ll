; RUN: llvm-link %s %p/Inputs/empty-objc.ll -S | FileCheck %s

; It tests whether Swift bitcode can be successfully linked with Objecitive-C bitcode.
; During the process, the IRUpgrader turns a i32 type "Objective-C Garbage Collection"
; into i8 value. If the higher bits are set, it adds the module flag for swift info.

target triple = "x86_64-apple-macosx10.15.0"

@__swift_reflection_version = linkonce_odr hidden constant i16 3
@llvm.used = appending global [1 x ptr] [ptr @__swift_reflection_version], section "llvm.metadata", align 8

define i32 @main(i32 %0, ptr %1) #0 {
  ret i32 0
}

attributes #0 = { "frame-pointer"="all" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!swift.module.flags = !{!9}
!llvm.linker.options = !{!10, !11, !12}
!llvm.asan.globals = !{!13}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 10, i32 15]}
!1 = !{i32 1, !"Objective-C Version", i32 2}
!2 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!3 = !{i32 1, !"Objective-C Image Info Section", !"__DATA,__objc_imageinfo,regular,no_dead_strip"}
!4 = !{i32 4, !"Objective-C Garbage Collection", i32 83953408}
!5 = !{i32 1, !"Objective-C Class Properties", i32 64}
!6 = !{i32 1, !"wchar_size", i32 4}
!7 = !{i32 7, !"PIC Level", i32 2}
!8 = !{i32 1, !"Swift Version", i32 7}
!9 = !{!"standard-library", i1 false}
!10 = !{!"-lswiftSwiftOnoneSupport"}
!11 = !{!"-lswiftCore"}
!12 = !{!"-lobjc"}
!13 = !{ptr @llvm.used, null, null, i1 false, i1 true}

; CHECK: !{{[0-9]+}} = !{i32 1, !"Objective-C Garbage Collection", i8 0}
; CHECK: !{{[0-9]+}} = !{i32 1, !"Swift ABI Version", i32 7}
; CHECK: !{{[0-9]+}} = !{i32 1, !"Swift Major Version", i8 5}
; CHECK: !{{[0-9]+}} = !{i32 1, !"Swift Minor Version", i8 1}
