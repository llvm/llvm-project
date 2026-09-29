; Round trip test for DW_OP_LLVM_implicit_pointer metadata

; RUN: llvm-as < %s | llvm-dis | FileCheck %s

;---------------------------
;static const char *b = "opq";
;volatile int v;
;int main() {
;  int var = 4;
;  int *ptr1;
;  int **ptr2;
;
;  v++;
;  ptr1 = &var;
;  ptr2 = &ptr1;
;  v++;
;
;  return *ptr1 - 5 + **ptr2;
;}
;---------------------------

; ModuleID = 'LLVM_implicit_pointer.c'
source_filename = "LLVM_implicit_pointer.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@v = dso_local global i32 0, align 4, !dbg !0

; Function Attrs: nofree norecurse nounwind uwtable
define dso_local i32 @main() local_unnamed_addr !dbg !12 {
entry:
; CHECK: #dbg_value(i32 4, [[VAR:![0-9]+]], !DIExpression(),
  call void @llvm.dbg.value(metadata i32 4, metadata !16, metadata !DIExpression()), !dbg !21
  %0 = load volatile i32, ptr @v, align 4, !dbg !22, !tbaa !23
  %inc = add nsw i32 %0, 1, !dbg !22
  store volatile i32 %inc, ptr @v, align 4, !dbg !22, !tbaa !23

; CHECK: #dbg_value(i32 4, [[PTR1:![0-9]+]], !DIExpression(DW_OP_LLVM_implicit_pointer),
  call void @llvm.dbg.value(metadata i32 4, metadata !17, metadata !DIExpression(DW_OP_LLVM_implicit_pointer)), !dbg !21

; CHECK: #dbg_value(i32 4, [[PTR2:![0-9]+]], !DIExpression(DW_OP_LLVM_implicit_pointer, DW_OP_LLVM_implicit_pointer),
  call void @llvm.dbg.value(metadata i32 4, metadata !19, metadata !DIExpression(DW_OP_LLVM_implicit_pointer, DW_OP_LLVM_implicit_pointer)), !dbg !21
  %1 = load volatile i32, ptr @v, align 4, !dbg !27, !tbaa !23
  %inc1 = add nsw i32 %1, 1, !dbg !27
  store volatile i32 %inc1, ptr @v, align 4, !dbg !27, !tbaa !23
  ret i32 3, !dbg !28
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9, !10}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "v", scope: !2, file: !3, line: 2, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 12.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "LLVM_implicit_pointer.c", directory: "/dir", checksumkind: CSK_MD5, checksum: "218aaa8dc9f04b056b56d944d06383dd")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !7)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{i32 7, !"Dwarf Version", i32 5}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{!"clang version 12.0.0"}
!12 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 3, type: !13, scopeLine: 3, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !15)
!13 = !DISubroutineType(types: !14)
!14 = !{!7}
!15 = !{!16, !17, !19}
; CHECK: [[VAR]] = !DILocalVariable(name: "var"
!16 = !DILocalVariable(name: "var", scope: !12, file: !3, line: 4, type: !7)
; CHECK: [[PTR1]] = !DILocalVariable(name: "ptr1"
!17 = !DILocalVariable(name: "ptr1", scope: !12, file: !3, line: 5, type: !18)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
; CHECK: [[PTR2]] = !DILocalVariable(name: "ptr2"
!19 = !DILocalVariable(name: "ptr2", scope: !12, file: !3, line: 6, type: !20)
!20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 64)
!21 = !DILocation(line: 0, scope: !12)
!22 = !DILocation(line: 8, column: 4, scope: !12)
!23 = !{!24, !24, i64 0}
!24 = !{!"int", !25, i64 0}
!25 = !{!"omnipotent char", !26, i64 0}
!26 = !{!"Simple C/C++ TBAA"}
!27 = !DILocation(line: 11, column: 4, scope: !12)
!28 = !DILocation(line: 13, column: 3, scope: !12)
