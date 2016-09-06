; RUN: opt -analyze -module-debuginfo < %s | FileCheck %s


; This module is generated from the following c-code:
;
; > union X;
; >
; > struct Y {
; >     union X *x;
; > };
; >
; > struct Y y;


; CHECK: Type: Y from /tmp/minimal.c:3 DW_TAG_structure_type
; CHECK: Type: x from /tmp/minimal.c:4 DW_TAG_member
; CHECK: Type: DW_TAG_pointer_type
; CHECK: Type: X from /tmp/minimal.c:1 DW_TAG_structure_type


%struct.Y = type { %struct.X* }
%struct.X = type opaque

@y = common global %struct.Y zeroinitializer, align 8

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.7.0 (http://llvm.org/git/clang.git 247b30a043eb8f39ea3708e7e995089da0a6b00f) (http://llvm.org/git/llvm.git 6ecc7365a89c771fd229bdd9ffcc178684ea1aa5)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !3, imports: !2)
!1 = !DIFile(filename: "minimal.c", directory: "/tmp")
!2 = !{}
!3 = !{!4}
!4 = !DIGlobalVariable(name: "y", scope: !0, file: !1, line: 7, type: !5, isLocal: false, isDefinition: true, variable: %struct.Y* @y)
!5 = !DICompositeType(tag: DW_TAG_structure_type, name: "Y", file: !1, line: 3, size: 64, align: 64, elements: !6)
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !5, file: !1, line: 4, baseType: !8, size: 64, align: 64)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64, align: 64)
!9 = !DICompositeType(tag: DW_TAG_structure_type, name: "X", file: !1, line: 1, flags: DIFlagFwdDecl)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{!"clang version 3.7.0 (http://llvm.org/git/clang.git 247b30a043eb8f39ea3708e7e995089da0a6b00f) (http://llvm.org/git/llvm.git 6ecc7365a89c771fd229bdd9ffcc178684ea1aa5)"}
