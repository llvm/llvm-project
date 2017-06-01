; RUN: llc -mtriple=x86_64-macosx %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; CHECK: DW_TAG_structure_type
; CHECK:                 DW_AT_declaration
; CHECK:                 DW_AT_APPLE_runtime_class

source_filename = "test/DebugInfo/X86/objc-fwd-decl.ll"

%0 = type opaque

@a = common global %0* null, align 8, !dbg !0

!llvm.dbg.cu = !{!5}
!llvm.module.flags = !{!8, !9, !10, !11, !12, !13}

!0 = !DIGlobalVariableExpression(var: !1)
!1 = !DIGlobalVariable(name: "a", scope: null, file: !2, line: 3, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "foo.m", directory: "/Users/echristo")
!3 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64, align: 64)
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "FooBarBaz", file: !2, line: 1, flags: DIFlagFwdDecl, runtimeLang: DW_LANG_ObjC)
!5 = distinct !DICompileUnit(language: DW_LANG_ObjC, file: !2, producer: "clang version 3.1 (trunk 152054 trunk 152094)", isOptimized: false, runtimeVersion: 2, emissionKind: FullDebug, enums: !6, retainedTypes: !6, globals: !7, imports: !6)
!6 = !{}
!7 = !{!0}
!8 = !{i32 1, !"Objective-C Version", i32 2}
!9 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!10 = !{i32 1, !"Objective-C Image Info Section", !"__DATA, __objc_imageinfo, regular, no_dead_strip"}
!11 = !{i32 4, !"Objective-C Garbage Collection", i32 0}
!12 = !{i32 1, !"Debug Info Version", i32 3}
!13 = !{i32 4, !"Objective-C Class Properties", i32 0}

