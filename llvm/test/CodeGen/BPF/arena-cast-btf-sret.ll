; RUN: opt -S -mtriple=bpfel -passes=bpf-arena-cast < %s | FileCheck %s

; A function that returns a struct in memory has one more argument than it has
; parameters in the source, and debug info counts the latter:
;   struct s g(char *k, long *h, void *c __attribute__((btf_decl_tag("arg:ctx"))));
; k gets to a helper, h is an address in arena, the source says what c is.

; CHECK: !DISubroutineType(types: ![[TYPES:[0-9]+]])
; CHECK: ![[TYPES]] = !{!{{[0-9]+}}, ![[K:[0-9]+]], ![[H:[0-9]+]], ![[C:[0-9]+]]}
; CHECK: ![[K]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !{{[0-9]+}}, size: 64)
; CHECK: ![[H]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !{{[0-9]+}}, size: 64, annotations: ![[TAGS:[0-9]+]])
; CHECK: ![[TAGS]] = !{![[TAG:[0-9]+]]}
; CHECK: ![[TAG]] = !{!"btf_type_tag", !"arena"}
; CHECK: ![[C]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)

define void @g(ptr sret([32 x i8]) %o, ptr %k, ptr %h, ptr %c) !dbg !4 {
  %r = call i64 inttoptr (i64 6 to ptr)(ptr %k, i32 2), !dbg !22
  %v = load i64, ptr %h, !dbg !22
  store i64 %v, ptr %o, !dbg !22
  ret void, !dbg !22
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/tmp")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "g", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !16)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !12, !14, !15}
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s", file: !1, line: 1, size: 256, elements: !8)
!8 = !{!9}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !7, file: !1, line: 1, baseType: !10, size: 256)
!10 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, size: 256, elements: !23)
!11 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!13 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!16 = !{!17, !18, !19}
!17 = !DILocalVariable(name: "k", arg: 1, scope: !4, file: !1, line: 1, type: !12)
!18 = !DILocalVariable(name: "h", arg: 2, scope: !4, file: !1, line: 1, type: !14)
!19 = !DILocalVariable(name: "c", arg: 3, scope: !4, file: !1, line: 1, type: !15, annotations: !20)
!20 = !{!21}
!21 = !{!"btf_decl_tag", !"arg:ctx"}
!22 = !DILocation(line: 1, column: 1, scope: !4)
!23 = !{!24}
!24 = !DISubrange(count: 4)
