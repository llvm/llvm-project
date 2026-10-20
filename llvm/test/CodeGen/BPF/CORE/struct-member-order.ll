; RUN: opt -O2 %s | llc -mtriple=bpfel -filetype=asm -o - | FileCheck %s

; The physical GEP index and DI member index differ because the DI elements are
; in source order while the struct is laid out in offset order. The immediate
; still comes from the original DI member, but the CO-RE access string must use
; that member's position in BTF. Members with equal offsets retain DI order.
;
; This mirrors rustc debug info for:
;
;   pub struct S {
;       pub a: (),
;       pub b: u32,
;       pub c: (),
;   }
;
; rustc emits the members in source order at offsets 32, 0, and 32 bits.

; CHECK-LABEL: test:
; CHECK:       [[TEST_RELOC:.Ltmp[0-9]+]]:
; CHECK-NEXT:  r2 = 4
; CHECK-LABEL: test_equal_offset:
; CHECK:       [[EQUAL_RELOC:.Ltmp[0-9]+]]:
; CHECK-NEXT:  r2 = 4

; CHECK:       .section        .BTF
; CHECK:       .long   [[#]]                    # BTF_KIND_STRUCT(id = [[STRUCT_ID:.*]])
; CHECK:       .ascii  ".text"                  # string offset=[[TEXT:.*]]
; CHECK:       .ascii  "0:1"                    # string offset=[[TEST_ACCESS:.*]]
; CHECK:       .ascii  "0:2"                    # string offset=[[EQUAL_ACCESS:.*]]

; CHECK:       .section        .BTF.ext
; CHECK:       .long   [[#]]                    # FieldReloc
; CHECK-NEXT:  .long   [[TEXT]]
; CHECK-NEXT:  .long   2
; CHECK-NEXT:  .long   [[TEST_RELOC]]
; CHECK-NEXT:  .long   [[STRUCT_ID]]
; CHECK-NEXT:  .long   [[TEST_ACCESS]]
; CHECK-NEXT:  .long   0
; CHECK-NEXT:  .long   [[EQUAL_RELOC]]
; CHECK-NEXT:  .long   [[STRUCT_ID]]
; CHECK-NEXT:  .long   [[EQUAL_ACCESS]]
; CHECK-NEXT:  .long   0

target triple = "bpf"

%struct.s = type { i32, [0 x i8], [0 x i8] }

define dso_local i32 @test(ptr %arg) local_unnamed_addr !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata ptr %arg, metadata !17, metadata !DIExpression()), !dbg !18
  %field = tail call ptr @llvm.preserve.struct.access.index.p0.p0.ss(ptr elementtype(%struct.s) %arg, i32 1, i32 0), !dbg !19, !llvm.preserve.access.index !12
  %call = tail call i32 @get_value(ptr %field), !dbg !20
  ret i32 %call, !dbg !21
}

define dso_local i32 @test_equal_offset(ptr %arg) local_unnamed_addr !dbg !23 {
entry:
  call void @llvm.dbg.value(metadata ptr %arg, metadata !24, metadata !DIExpression()), !dbg !25
  %field = tail call ptr @llvm.preserve.struct.access.index.p0.p0.ss(ptr elementtype(%struct.s) %arg, i32 2, i32 2), !dbg !26, !llvm.preserve.access.index !12
  %call = tail call i32 @get_value(ptr %field), !dbg !27
  ret i32 %call, !dbg !28
}

declare dso_local i32 @get_value(ptr) local_unnamed_addr

declare ptr @llvm.preserve.struct.access.index.p0.p0.ss(ptr, i32 immarg, i32 immarg)

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !1, producer: "rustc", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "test.rs", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"rustc"}
!7 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !16)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11}
!10 = !DIBasicType(name: "u32", size: 32, encoding: DW_ATE_unsigned)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s", file: !1, line: 1, size: 32, elements: !13)
!13 = !{!15, !14, !22}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !12, file: !1, line: 1, baseType: !10, size: 32)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !12, file: !1, line: 1, baseType: !30, offset: 32)
!16 = !{!17}
!17 = !DILocalVariable(name: "arg", arg: 1, scope: !7, file: !1, line: 1, type: !11)
!18 = !DILocation(line: 0, scope: !7)
!19 = !DILocation(line: 1, column: 1, scope: !7)
!20 = !DILocation(line: 1, column: 1, scope: !7)
!21 = !DILocation(line: 1, column: 1, scope: !7)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !12, file: !1, line: 1, baseType: !30, offset: 32)
!23 = distinct !DISubprogram(name: "test_equal_offset", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !29)
!24 = !DILocalVariable(name: "arg", arg: 1, scope: !23, file: !1, line: 2, type: !11)
!25 = !DILocation(line: 0, scope: !23)
!26 = !DILocation(line: 2, column: 1, scope: !23)
!27 = !DILocation(line: 2, column: 1, scope: !23)
!28 = !DILocation(line: 2, column: 1, scope: !23)
!29 = !{!24}
!30 = !DIBasicType(name: "()", encoding: DW_ATE_unsigned)
