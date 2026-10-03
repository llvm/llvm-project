; Argument tracing across ABI lowering. A by-value struct the ABI passed in
; registers has no address, so it is reported as one call per register piece,
; all carrying the same source-level parameter index. A struct the ABI passed
; indirectly does have an address, and is reported through it with its field
; offset table.
;
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=3 -sanitizer-coverage-trace-args -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.big = type { i64, i64, i64, i64, i64 }

; Only struct big needs a field table: it is the only one reported by address.
; CHECK: @__sancov_offsets_ = private unnamed_addr constant [10 x i64] [i64 0, i64 8, i64 8, i64 8, i64 16, i64 8, i64 24, i64 8, i64 32, i64 8]

; int use_pair(struct pair p, int z) with struct pair { int x; int y; } coerced
; into a single i64 and split back into two i32 fragments. Both pieces are
; reported under index 0, and `z` keeps index 1.
define i32 @use_pair(i64 %0, i32 %1) !dbg !13 {
entry:
  %2 = trunc i64 %0 to i32
  %3 = lshr i64 %0, 32
  %4 = trunc nuw i64 %3 to i32
    #dbg_value(i32 %2, !17, !DIExpression(DW_OP_LLVM_fragment, 0, 32), !19)
    #dbg_value(i32 %4, !17, !DIExpression(DW_OP_LLVM_fragment, 32, 32), !19)
    #dbg_value(i32 %1, !18, !DIExpression(), !19)
  %5 = add i32 %2, %4
  ret i32 %5
}
; Nothing is spilled, so no field table is built for struct pair either.
; CHECK-LABEL: define i32 @use_pair(
; CHECK-NOT: alloca
; CHECK: %[[LO:[0-9]+]] = zext i32 %2 to i64
; CHECK: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @use_pair to i64), i32 0, i32 4, i64 %[[LO]], ptr null, i32 0)
; CHECK: %[[HI:[0-9]+]] = zext i32 %4 to i64
; CHECK: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @use_pair to i64), i32 0, i32 4, i64 %[[HI]], ptr null, i32 0)
; CHECK: %[[Z:[0-9]+]] = zext i32 %1 to i64
; CHECK: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @use_pair to i64), i32 1, i32 4, i64 %[[Z]], ptr null, i32 0)

; A fragment whose location is an expression that *computes* the value - here a
; field shifted out of a wider register - is not a value that can be reported,
; so it is dropped while the fragments that are values are still reported.
;
; The dropped fragment is also defined before the one that follows it, so this
; doubles as a check that the calls are placed after every value they report:
; opt runs the verifier, which rejects a use that does not dominate.
define i32 @use_trio([2 x i64] %0) !dbg !30 {
entry:
  %1 = extractvalue [2 x i64] %0, 0
  %2 = trunc i64 %1 to i32
  %3 = extractvalue [2 x i64] %0, 1
    #dbg_value(i32 %2, !34, !DIExpression(DW_OP_LLVM_fragment, 0, 32), !35)
    #dbg_value(i64 %1, !34, !DIExpression(DW_OP_constu, 32, DW_OP_shr, DW_OP_LLVM_convert, 64, DW_ATE_unsigned, DW_OP_LLVM_convert, 32, DW_ATE_unsigned, DW_OP_stack_value, DW_OP_LLVM_fragment, 32, 32), !35)
    #dbg_value(i64 %3, !34, !DIExpression(DW_OP_LLVM_fragment, 64, 64), !35)
  ret i32 %2
}
; CHECK-LABEL: define i32 @use_trio(
; CHECK: %[[A:[0-9]+]] = zext i32 %2 to i64
; CHECK: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @use_trio to i64), i32 0, i32 4, i64 %[[A]], ptr null, i32 0)
; CHECK: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @use_trio to i64), i32 0, i32 8, i64 %3, ptr null, i32 0)
; CHECK-NOT: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @use_trio to i64), i32 1

; void use_big(struct big b) passed indirectly: the parameter is a pointer to
; the caller's copy, described by the struct's own type, so it is reported as
; the address of that copy with its five fields.
define void @use_big(ptr byval(%struct.big) align 8 %0) !dbg !20 {
entry:
    #dbg_value(ptr %0, !24, !DIExpression(), !25)
  ret void
}
; CHECK-LABEL: define void @use_big(
; CHECK: %[[ADDR:[0-9]+]] = ptrtoint ptr %0 to i64
; CHECK: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @use_big to i64), i32 0, i32 40, i64 %[[ADDR]], ptr @__sancov_offsets_, i32 5)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1, !2}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, isOptimized: true, emissionKind: FullDebug)
!1 = !{i32 2, !"Dwarf Version", i32 5}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DIFile(filename: "abi.c", directory: "/")
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)

; struct pair { int x; int y; }
!7 = !DICompositeType(tag: DW_TAG_structure_type, name: "pair", file: !3, size: 64, elements: !8)
!8 = !{!9, !10}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !7, file: !3, baseType: !4, size: 32)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !7, file: !3, baseType: !4, size: 32, offset: 32)

; struct big { long a, b, c, d, e; }
!11 = !DICompositeType(tag: DW_TAG_structure_type, name: "big", file: !3, size: 320, elements: !12)
!12 = !{!41, !42, !43, !44, !45}

!13 = distinct !DISubprogram(name: "use_pair", scope: !3, file: !3, line: 1, type: !14, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !16)
!14 = !DISubroutineType(types: !15)
!15 = !{!4, !7, !4}
!16 = !{!17, !18}
!17 = !DILocalVariable(name: "p", arg: 1, scope: !13, file: !3, line: 1, type: !7)
!18 = !DILocalVariable(name: "z", arg: 2, scope: !13, file: !3, line: 1, type: !4)
!19 = !DILocation(line: 1, column: 1, scope: !13)

!20 = distinct !DISubprogram(name: "use_big", scope: !3, file: !3, line: 6, type: !21, scopeLine: 6, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !23)
!21 = !DISubroutineType(types: !22)
!22 = !{null, !11}
!23 = !{!24}
!24 = !DILocalVariable(name: "b", arg: 1, scope: !20, file: !3, line: 6, type: !11)
!25 = !DILocation(line: 6, column: 1, scope: !20)

; struct trio { int a; int b; long c; }
!29 = !DICompositeType(tag: DW_TAG_structure_type, name: "trio", file: !3, size: 128, elements: !39)
!30 = distinct !DISubprogram(name: "use_trio", scope: !3, file: !3, line: 11, type: !31, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !33)
!31 = !DISubroutineType(types: !32)
!32 = !{!4, !29}
!33 = !{!34}
!34 = !DILocalVariable(name: "t", arg: 1, scope: !30, file: !3, line: 11, type: !29)
!35 = !DILocation(line: 11, column: 1, scope: !30)
!39 = !{!36, !37, !38}
!36 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !29, file: !3, baseType: !4, size: 32)
!37 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !29, file: !3, baseType: !4, size: 32, offset: 32)
!38 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !29, file: !3, baseType: !5, size: 64, offset: 64)
!41 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !11, file: !3, baseType: !5, size: 64)
!42 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !11, file: !3, baseType: !5, size: 64, offset: 64)
!43 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !11, file: !3, baseType: !5, size: 64, offset: 128)
!44 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !11, file: !3, baseType: !5, size: 64, offset: 192)
!45 = !DIDerivedType(tag: DW_TAG_member, name: "e", scope: !11, file: !3, baseType: !5, size: 64, offset: 256)
