; RUN: llc -mtriple=x86_64-macosx %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; CHECK: DW_TAG_structure_type
; CHECK:                 DW_AT_declaration
; CHECK:                 DW_AT_APPLE_runtime_class

%0 = type opaque

@a = common global %0* null, align 8

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10, !11, !12, !14}

!0 = metadata !{i32 786449, metadata !13, i32 16, metadata !"clang version 3.1 (trunk 152054 trunk 152094)", i1 false, metadata !"", i32 2, metadata !1, metadata !1, metadata !1, metadata !3,  metadata !3, metadata !""} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{i32 786484, i32 0, null, metadata !"a", metadata !"a", metadata !"", metadata !6, i32 3, metadata !7, i32 0, i32 1, %0** @a, null} ; [ DW_TAG_variable ]
!6 = metadata !{i32 786473, metadata !13} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !8} ; [ DW_TAG_pointer_type ]
!8 = metadata !{i32 786451, metadata !13, null, metadata !"FooBarBaz", i32 1, i32 0, i32 0, i32 0, i32 4, null, null, i32 16, null, null, null} ; [ DW_TAG_structure_type ] [FooBarBaz] [line 1, size 0, align 0, offset 0] [decl] [from ]
!9 = metadata !{i32 1, metadata !"Objective-C Version", i32 2}
!10 = metadata !{i32 1, metadata !"Objective-C Image Info Version", i32 0}
!11 = metadata !{i32 1, metadata !"Objective-C Image Info Section", metadata !"__DATA, __objc_imageinfo, regular, no_dead_strip"}
!12 = metadata !{i32 4, metadata !"Objective-C Garbage Collection", i32 0}
!13 = metadata !{metadata !"foo.m", metadata !"/Users/echristo"}
!14 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
