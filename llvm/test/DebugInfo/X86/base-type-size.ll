; RUN: llc %s --filetype=obj -o - | llvm-dwarfdump - -o - | FileCheck %s

;; cat test.cpp
;; void ext(bool);
;; void fun(bool b) { ext(b); }
;; $ clang++ test.cpp -o - -emit-llvm -S -O2 -gdwarf-5
;;
;; Check that the DW_TAG_base_type DIE for the 1u conversion in the DIExpression
;; has a non-zero DW_AT_byte_size attribute.

; CHECK: DW_TAG_base_type
; CHECK-NEXT: DW_AT_name      ("DW_ATE_unsigned_1")
; CHECK-NEXT: DW_AT_encoding  (DW_ATE_unsigned)
; CHECK-NEXT: DW_AT_byte_size (0x01)

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local void @_Z3funb(i1 zeroext %b) local_unnamed_addr #0 !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i1 %b, metadata !12, metadata !DIExpression(DW_OP_LLVM_convert, 1, DW_ATE_unsigned, DW_OP_LLVM_convert, 8, DW_ATE_unsigned, DW_OP_stack_value)), !dbg !13
  tail call void @_Z3extb(i1 zeroext %b), !dbg !14
  ret void, !dbg !15
}

declare !dbg !16 dso_local void @_Z3extb(i1 zeroext) local_unnamed_addr
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"uwtable", i32 1}
!6 = !{!"clang version 14.0.0"}
!7 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funb", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!11 = !{!12}
!12 = !DILocalVariable(name: "b", arg: 1, scope: !7, file: !1, line: 2, type: !10)
!13 = !DILocation(line: 0, scope: !7)
!14 = !DILocation(line: 2, column: 20, scope: !7)
!15 = !DILocation(line: 2, column: 28, scope: !7)
!16 = !DISubprogram(name: "ext", linkageName: "_Z3extb", scope: !1, file: !1, line: 1, type: !8, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !17)
!17 = !{}
