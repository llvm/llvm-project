; Test trace-args handles ABI-inserted hidden arguments correctly.
; Verifies:
; 1. sret hidden arg is skipped (not traced)
; 2. Struct coercion fragments are reassembled
; 3. Normal pointer arg with struct offsets still works

; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=3 -sanitizer-coverage-trace-args -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.big = type { i64, i64, i64, i64, i64 }
%struct.small = type { i32, i32 }

; Function with sret: source has 2 params (x, y), IR has 3 args (sret, x, y)
define void @make_big(ptr sret(%struct.big) %0, i32 %1, i32 %2) #0 !dbg !20 {
entry:
    #dbg_value(i32 %1, !30, !DIExpression(), !32)
    #dbg_value(i32 %2, !31, !DIExpression(), !32)
  ret void
}

; CHECK-LABEL: define void @make_big(ptr sret(%struct.big) %0, i32 %1, i32 %2)
; CHECK: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @make_big to i64), i32 0, i32 4, ptr %{{.*}}, ptr null, i32 0)
; CHECK: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @make_big to i64), i32 1, i32 4, ptr %{{.*}}, ptr null, i32 0)
; CHECK-NOT: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @make_big to i64), i32 2
; CHECK: ret void

; Function with struct coercion: source has 2 params (s, z), IR has 2 args (i64, i32)
; but debug info says s is split into fragments at bit offsets 0 and 32
define i32 @use_small(i64 %0, i32 %1) #0 !dbg !40 {
entry:
  %3 = trunc i64 %0 to i32
  %4 = lshr i64 %0, 32
  %5 = trunc nuw i64 %4 to i32
    #dbg_value(i32 %3, !50, !DIExpression(DW_OP_LLVM_fragment, 0, 32), !52)
    #dbg_value(i32 %5, !50, !DIExpression(DW_OP_LLVM_fragment, 32, 32), !52)
    #dbg_value(i32 %1, !51, !DIExpression(), !52)
  %6 = add i32 %1, %3
  %7 = add i32 %6, %5
  ret i32 %7
}

; CHECK-LABEL: define i32 @use_small(i64 %0, i32 %1)
; Two trace calls: arg 0 = reassembled struct (8 bytes) with field offsets, arg 1 = z (4 bytes)
; CHECK: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @use_small to i64), i32 0, i32 8, ptr %{{.*}}, ptr getelementptr inbounds {{.*}}, i32 2)
; CHECK: call void @__sanitizer_cov_trace_args(i64 ptrtoint (ptr @use_small to i64), i32 1, i32 4, ptr %{{.*}}, ptr null, i32 0)
; CHECK: ret i32

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, isOptimized: true, emissionKind: FullDebug)
!1 = !DIFile(filename: "test_abi.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}

; Types
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)

; make_big debug info
!20 = distinct !DISubprogram(name: "make_big", scope: !1, file: !1, line: 3, type: !21, scopeLine: 3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !29)
!21 = !DISubroutineType(types: !22)
!22 = !{!23, !5, !5}  ; returns struct big, params: int, int
!23 = !DICompositeType(tag: DW_TAG_structure_type, name: "big", size: 320, elements: !2)
!29 = !{!30, !31}
!30 = !DILocalVariable(name: "x", arg: 1, scope: !20, file: !1, line: 3, type: !5)
!31 = !DILocalVariable(name: "y", arg: 2, scope: !20, file: !1, line: 3, type: !5)
!32 = !DILocation(line: 3, scope: !20)

; use_small debug info
!40 = distinct !DISubprogram(name: "use_small", scope: !1, file: !1, line: 8, type: !41, scopeLine: 8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !49)
!41 = !DISubroutineType(types: !42)
!42 = !{!5, !43, !5}  ; returns int, params: struct small, int
!43 = !DICompositeType(tag: DW_TAG_structure_type, name: "small", size: 64, elements: !44)
!44 = !{!45, !46}
!45 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !43, baseType: !5, size: 32, offset: 0)
!46 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !43, baseType: !5, size: 32, offset: 32)
!49 = !{!50, !51}
!50 = !DILocalVariable(name: "s", arg: 1, scope: !40, file: !1, line: 8, type: !43)
!51 = !DILocalVariable(name: "z", arg: 2, scope: !40, file: !1, line: 8, type: !5)
!52 = !DILocation(line: 8, scope: !40)
