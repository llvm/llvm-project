; RUN: llc %s -start-after=codegenprepare -stop-before=finalize-isel -experimental-debug-variable-locations=false -o - | FileCheck %s

;; Generated from clang -O2 -emit-llvm -S -g reduce.c -o -
;; $ cat reduce.c
;; struct {
;;   int a;
;; } * b;
;; int c;
;; void d() {
;;   int *e = &b->a - 1;  // XXX
;;   c = *e;
;; }
;;
;; The line marked XXX becomes a load and gep in IR. We have a variadic
;; dbg.value using the gep, but we lose that gep in SelectionDAG. Ensure that
;; we salvage the value.

; CHECK: [[E_REG:%[0-9]+]]{{.+}} = MOV{{.+}} @b
; CHECK: DBG_VALUE_LIST {{.*}}, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_constu, 4, DW_OP_minus, DW_OP_stack_value), [[E_REG]], debug-location

target triple = "x86_64-unknown-linux-gnu"

%struct.anon = type { i32 }

@b = dso_local local_unnamed_addr global ptr null, align 8, !dbg !0
@c = dso_local local_unnamed_addr global i32 0, align 4, !dbg !6

define dso_local void @d() local_unnamed_addr !dbg !17 {
entry:
  %0 = load ptr, ptr @b, align 8, !dbg !23
  %add.ptr = getelementptr inbounds %struct.anon, ptr %0, i64 -1, i32 0, !dbg !28
  call void @llvm.dbg.value(metadata !DIArgList(ptr %add.ptr), metadata !21, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_stack_value)), !dbg !29
  %1 = load i32, ptr %add.ptr, align 4, !dbg !30
  store i32 %1, ptr @c, align 4, !dbg !33
  ret void, !dbg !34
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!13, !14, !15}
!llvm.ident = !{!16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !3, line: 3, type: !9, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 11.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "reduce.c", directory: "/")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "c", scope: !2, file: !3, line: 4, type: !8, isLocal: false, isDefinition: true)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !3, line: 1, size: 32, elements: !11)
!11 = !{!12}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !10, file: !3, line: 2, baseType: !8, size: 32)
!13 = !{i32 7, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"wchar_size", i32 4}
!16 = !{!"clang version 11.0.0"}
!17 = distinct !DISubprogram(name: "d", scope: !3, file: !3, line: 5, type: !18, scopeLine: 5, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !20)
!18 = !DISubroutineType(types: !19)
!19 = !{null}
!20 = !{!21}
!21 = !DILocalVariable(name: "e", scope: !17, file: !3, line: 6, type: !22)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!23 = !DILocation(line: 6, column: 13, scope: !17)
!28 = !DILocation(line: 6, column: 18, scope: !17)
!29 = !DILocation(line: 0, scope: !17)
!30 = !DILocation(line: 7, column: 7, scope: !17)
!33 = !DILocation(line: 7, column: 5, scope: !17)
!34 = !DILocation(line: 8, column: 1, scope: !17)
