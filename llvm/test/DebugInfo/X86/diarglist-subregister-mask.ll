; RUN: llc -mtriple=x86_64-pc-linux-gnu -dwarf-version=5 -filetype=obj -o - %s \
; RUN:   | llvm-dwarfdump -debug-info - | FileCheck %s

;; Check that masking a subregister location for one DW_OP_LLVM_arg does not
;; leak into a following full-register DW_OP_LLVM_arg in the same expression.

; CHECK: DW_AT_location (DW_OP_breg5 RDI+0, DW_OP_constu 0xffffffff, DW_OP_and, DW_OP_convert {{.*}}"DW_ATE_signed_32", DW_OP_convert {{.*}}"DW_ATE_signed_64", DW_OP_breg4 RSI+0, DW_OP_eq, DW_OP_convert {{.*}}"DW_ATE_unsigned_1", DW_OP_convert {{.*}}"DW_ATE_unsigned_32", DW_OP_stack_value)
; CHECK-NEXT: DW_AT_name ("x")

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

@sinki = global i32 0, align 4
@sinku64 = global i64 0, align 8

define i32 @test(i32 %a, i64 %b) !dbg !7 {
entry:
    #dbg_value(i32 %a, !14, !DIExpression(), !17)
    #dbg_value(i64 %b, !15, !DIExpression(), !17)
    #dbg_value(!DIArgList(i32 %a, i64 %b), !16, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_convert, 32, DW_ATE_signed, DW_OP_LLVM_convert, 64, DW_ATE_signed, DW_OP_LLVM_arg, 1, DW_OP_eq, DW_OP_LLVM_convert, 1, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_stack_value), !17)
  %0 = load volatile i32, ptr @sinki, align 4
  %add = add i32 %0, %a, !dbg !18
  store volatile i32 %add, ptr @sinki, align 4, !dbg !19
  %1 = load volatile i64, ptr @sinku64, align 8, !dbg !20
  %add2 = add i64 %1, %b, !dbg !21
  store volatile i64 %add2, ptr @sinku64, align 8, !dbg !22
  ret i32 0, !dbg !23
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !24}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "test", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "diarglist-subregister-mask.c", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!7 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13, keyInstructions: true)
!8 = !DISubroutineType(types: !9)
!9 = !{!5, !10, !6}
!10 = !DIDerivedType(tag: DW_TAG_typedef, name: "int32_t", file: !1, line: 1, baseType: !5)
!11 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint64_t", file: !1, line: 1, baseType: !6)
!12 = !{}
!13 = !{!14, !15, !16}
!14 = !DILocalVariable(name: "a", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!15 = !DILocalVariable(name: "b", arg: 2, scope: !7, file: !1, line: 1, type: !11)
!16 = !DILocalVariable(name: "x", scope: !7, file: !1, line: 2, type: !5)
!17 = !DILocation(line: 0, scope: !7)
!18 = !DILocation(line: 3, column: 9, scope: !7, atomGroup: 1, atomRank: 2)
!19 = !DILocation(line: 3, column: 9, scope: !7, atomGroup: 1, atomRank: 1)
!20 = !DILocation(line: 4, column: 11, scope: !7)
!21 = !DILocation(line: 4, column: 11, scope: !7, atomGroup: 2, atomRank: 2)
!22 = !DILocation(line: 4, column: 11, scope: !7, atomGroup: 2, atomRank: 1)
!23 = !DILocation(line: 5, column: 3, scope: !7, atomGroup: 3, atomRank: 1)
!24 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
