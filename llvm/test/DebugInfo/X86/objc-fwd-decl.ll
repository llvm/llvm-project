; RUN: llc -mtriple=x86_64-macosx %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; CHECK: DW_TAG_structure_type
; CHECK:                 DW_AT_declaration
; CHECK:                 DW_AT_APPLE_runtime_class

%0 = type opaque

@a = common global %0* null, align 8

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9, !10, !11, !12, !14}

!0 = distinct !DICompileUnit(language: DW_LANG_ObjC, producer: "clang version 3.1 (trunk 152054 trunk 152094)", isOptimized: false, runtimeVersion: 2, emissionKind: 0, file: !13, enums: !1, retainedTypes: !1, subprograms: !1, globals: !3, imports:  !1)
!1 = !{}
!3 = !{!5}
!5 = !DIGlobalVariable(name: "a", line: 3, isLocal: false, isDefinition: true, scope: null, file: !6, type: !7, variable: %0** @a)
!6 = !DIFile(filename: "foo.m", directory: "/Users/echristo")
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !8)
!8 = !DICompositeType(tag: DW_TAG_structure_type, name: "FooBarBaz", line: 1, flags: DIFlagFwdDecl, runtimeLang: DW_LANG_ObjC, file: !13)
!9 = !{i32 1, !"Objective-C Version", i32 2}
!10 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!11 = !{i32 1, !"Objective-C Image Info Section", !"__DATA, __objc_imageinfo, regular, no_dead_strip"}
!12 = !{i32 4, !"Objective-C Garbage Collection", i32 0}
!13 = !DIFile(filename: "foo.m", directory: "/Users/echristo")
!14 = !{i32 1, !"Debug Info Version", i32 3}
