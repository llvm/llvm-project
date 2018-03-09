; RUN: llc -filetype=asm %s -o - | FileCheck %s
; Test large integral function arguments passed in multiple registers.
;
; Generated from (-Os):
;
; signed long long g;
; void write(signed long long offset);
; signed long long foo(signed long long offset) {
;   if (offset != g)
;     write(offset);
;   return offset;
; }
source_filename = "test.ii"
target datalayout = "e-m:o-p:32:32-i64:64-a:0:32-n32-S128"
target triple = "thumbv7k-apple-watchos2.0.0"

@g = local_unnamed_addr global i64 0, align 8, !dbg !0

; Function Attrs: optsize ssp
define i64 @_Z3foox(i64 returned) local_unnamed_addr #0 !dbg !13 {
  tail call void @llvm.dbg.value(metadata i64 %0, metadata !17, metadata !DIExpression()), !dbg !18
  ; CHECK: @DEBUG_VALUE: foo:offset <- [DW_OP_LLVM_fragment 0 32] %r5
  ; CHECK: @DEBUG_VALUE: foo:offset <- [DW_OP_LLVM_fragment 32 32] %r4

  %2 = load i64, i64* @g, align 8, !dbg !19, !tbaa !21
  %3 = icmp eq i64 %2, %0, !dbg !19
  br i1 %3, label %5, label %4, !dbg !25

; <label>:4:                                      ; preds = %1
  tail call void @_Z5writex(i64 %0) #3, !dbg !26
  br label %5, !dbg !26

; <label>:5:                                      ; preds = %1, %4
  ret i64 %0, !dbg !27
}

; Function Attrs: optsize
declare void @_Z5writex(i64) local_unnamed_addr #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { optsize ssp }
attributes #1 = { optsize }
attributes #2 = { nounwind readnone speculatable }
attributes #3 = { optsize }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10, !11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 6.0.0 (trunk 312148) (llvm/trunk 312165)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "test.ii", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "long long int", size: 64, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{i32 1, !"min_enum_size", i32 4}
!11 = !{i32 7, !"PIC Level", i32 2}
!13 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foox", scope: !3, file: !3, line: 3, type: !14, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !2, variables: !16)
!14 = !DISubroutineType(types: !15)
!15 = !{!6, !6}
!16 = !{!17}
!17 = !DILocalVariable(name: "offset", arg: 1, scope: !13, file: !3, line: 3, type: !6)
!18 = !DILocation(line: 3, scope: !13)
!19 = !DILocation(line: 4, scope: !20)
!20 = distinct !DILexicalBlock(scope: !13, file: !3, line: 4)
!21 = !{!22, !22, i64 0}
!22 = !{!"long long", !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C++ TBAA"}
!25 = !DILocation(line: 4, scope: !13)
!26 = !DILocation(line: 5, scope: !20)
!27 = !DILocation(line: 6, scope: !13)
