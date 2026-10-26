; RUN: opt -S -mtriple=bpfel -passes=bpf-arena-data < %s | FileCheck %s

; The kernel manages a field that has a type tag, like __kptr_untrusted: it finds
; the field in the BTF of a map value, and what the program loads from there is
; a pointer for the verifier. Such a variable does not move. The tag "arena" is
; the one that the compiler writes itself.

; CHECK: @managed = dso_local local_unnamed_addr global ptr null, align 8, !dbg
; CHECK: @in_struct = dso_local local_unnamed_addr global %struct.anon zeroinitializer, align 8, !dbg
; CHECK: @in_arena = dso_local global ptr null, section ".bss.addr_space.1", align 8, !dbg

%struct.anon = type { i64, ptr }
@managed = dso_local local_unnamed_addr global ptr null, align 8, !dbg !0
@in_struct = dso_local local_unnamed_addr global %struct.anon zeroinitializer, align 8, !dbg !5
@in_arena = dso_local local_unnamed_addr global ptr null, align 8, !dbg !17
!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!22, !23, !24, !25}
!llvm.errno.tbaa = !{!27}
!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "managed", scope: !2, file: !7, line: 2, type: !13, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/")
!4 = !{!0, !5, !17}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "in_struct", scope: !2, file: !7, line: 3, type: !8, isLocal: false, isDefinition: true)
!7 = !DIFile(filename: "t.c", directory: "/")
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !7, line: 3, size: 128, elements: !9)
!9 = !{!10, !12}
!10 = !DIDerivedType(tag: DW_TAG_member, name: "cnt", scope: !8, file: !7, line: 3, baseType: !11, size: 64)
!11 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!12 = !DIDerivedType(tag: DW_TAG_member, name: "t", scope: !8, file: !7, line: 3, baseType: !13, size: 64, offset: 64)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64, annotations: !15)
!14 = !DICompositeType(tag: DW_TAG_structure_type, name: "task", file: !7, line: 1, flags: DIFlagFwdDecl)
!15 = !{!16}
!16 = !{!"btf_type_tag", !"kptr_untrusted"}
!17 = !DIGlobalVariableExpression(var: !18, expr: !DIExpression())
!18 = distinct !DIGlobalVariable(name: "in_arena", scope: !2, file: !7, line: 4, type: !19, isLocal: false, isDefinition: true)
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64, annotations: !20)
!20 = !{!21}
!21 = !{!"btf_type_tag", !"arena"}
!22 = !{i32 7, !"Dwarf Version", i32 5}
!23 = !{i32 2, !"Debug Info Version", i32 3}
!24 = !{i32 2, !"debugger-tuning", !"gdb"}
!25 = !{i32 7, !"frame-pointer", i32 2}
!27 = !{!28, !29, i64 0}
!28 = !{!"__libc_errno", !29, i64 0}
!29 = !{!"int", !30, i64 0}
!30 = !{!"omnipotent char", !31, i64 0}
!31 = !{!"Simple C/C++ TBAA"}
