; RUN: %llc_dwarf -filetype=obj -o %t.o < %s
; RUN: llvm-dwarfdump %t.o | FileCheck %s

;; Checks that when we have a set of debug values that are all different but
;; result in identical DW_AT_location entries, we are able to merge them into a
;; single DW_AT_location instead of producing a loclist with identical locations
;; for each.

;; Checks that we can merge a non-variadic debug value with an equivalent
;; variadic debug value.
; CHECK: DW_AT_location
; CHECK-SAME: (DW_OP_fbreg +{{[0-9]+}}, DW_OP_deref, DW_OP_stack_value)
; CHECK-NEXT: DW_AT_name    ("Var2")

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"struct.std::_Rb_tree_node_base" = type { i32, ptr, ptr, ptr }

define i32 @_ZN4llvm9MCContext12GetDwarfFileENS_9StringRefES1_jj(ptr %this, ptr %A, i64 %B, ptr %C, i64 %D, i32 %E, i32 %CUID, i1 %cmp.i.i.i.i.i) !dbg !10 {
entry:
  %CUID.addr = alloca i32, align 4
  store i32 %CUID, ptr %CUID.addr, align 4
  call void @llvm.dbg.value(metadata ptr %CUID.addr, metadata !26, metadata !DIExpression(DW_OP_deref, DW_OP_stack_value)), !dbg !17
  %0 = load ptr, ptr null, align 8, !dbg !18
  call void @llvm.dbg.value(metadata !DIArgList(ptr %CUID.addr), metadata !26, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_deref, DW_OP_stack_value)), !dbg !18
  br label %while.body.i.i.i.i

while.body.i.i.i.i:                               ; preds = %while.body.i.i.i.i, %entry
  %__x.addr.011.i.i.i.i = phi ptr [ %__x.addr.1.in.i.i.i.i, %while.body.i.i.i.i ], [ %0, %entry ]
  %_M_right.i.i.i.i.i = getelementptr inbounds %"struct.std::_Rb_tree_node_base", ptr %__x.addr.011.i.i.i.i, i64 0, i32 3, !dbg !20
  %__x.addr.1.in.i.i.i.i = select i1 %cmp.i.i.i.i.i, ptr %__x.addr.011.i.i.i.i, ptr null, !dbg !20
  br label %while.body.i.i.i.i
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 16.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang version 16.0.0"}
!10 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooPi", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !15)
!11 = !DISubroutineType(types: !12)
!12 = !{!13, !14}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!15 = !{}
!16 = !DILocalVariable(name: "Var1", scope: !10, file: !1, line: 1, type: !14)
!17 = !DILocation(line: 1, column: 14, scope: !10)
!18 = !DILocation(line: 2, column: 3, scope: !10)
!19 = !DILocation(line: 2, column: 5, scope: !10)
!20 = !DILocation(line: 3, column: 10, scope: !10)
!21 = !DILocation(line: 3, column: 9, scope: !10)
!22 = !DILocation(line: 3, column: 2, scope: !10)
!26 = !DILocalVariable(name: "Var2", scope: !10, file: !1, line: 1, type: !14)
