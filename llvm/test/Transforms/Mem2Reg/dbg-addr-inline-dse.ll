; RUN: opt -passes=mem2reg -S < %s | FileCheck %s -implicit-check-not="call void @llvm.dbg.addr"

; This example is intended to simulate this pass pipeline, which may not exist
; in practice:
; 1. DSE f from the original C source
; 2. Inline escape
; 3. mem2reg
; This exercises the corner case of multiple llvm.dbg.addr intrinsics.

; C source:
;
; void escape(int *px) { ++*px; }
; extern int global;
; void f(int x) {
;   escape(&x);
;   x = 1; // DSE should delete and insert dbg.value(i32 1)
;   global = x;
;   x = 2; // DSE should insert dbg.addr
;   escape(&x);
; }

; ModuleID = 'dse.c'
source_filename = "dse.c"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24215"

declare void @llvm.dbg.addr(metadata, metadata, metadata) #2
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

@global = external global i32, align 4

; Function Attrs: nounwind uwtable
define void @f(i32 %x) #0 !dbg !8 {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  call void @llvm.dbg.addr(metadata ptr %x.addr, metadata !13, metadata !DIExpression()), !dbg !18
  %ld.1 = load i32, ptr %x.addr, align 4, !dbg !19
  %inc.1 = add nsw i32 %ld.1, 1, !dbg !19
  store i32 %inc.1, ptr %x.addr, align 4, !dbg !19
  call void @llvm.dbg.value(metadata i32 1, metadata !13, metadata !DIExpression()), !dbg !20
  store i32 1, ptr @global, align 4, !dbg !22
  call void @llvm.dbg.addr(metadata ptr %x.addr, metadata !13, metadata !DIExpression()), !dbg !23
  store i32 2, ptr %x.addr, align 4, !dbg !23
  %ld.2 = load i32, ptr %x.addr, align 4, !dbg !19
  %inc.2 = add nsw i32 %ld.2, 1, !dbg !19
  store i32 %inc.2, ptr %x.addr, align 4, !dbg !19
  ret void, !dbg !25
}

; CHECK-LABEL: define void @f(i32 %x)
; CHECK: call void @llvm.dbg.value(metadata i32 %x, metadata !13, metadata !DIExpression())
; CHECK: %inc.1 = add nsw i32 %x, 1
; CHECK: call void @llvm.dbg.value(metadata i32 %inc.1, metadata !13, metadata !DIExpression())
; CHECK: call void @llvm.dbg.value(metadata i32 1, metadata !13, metadata !DIExpression())
; CHECK: store i32 1, ptr @global, align 4
; CHECK: call void @llvm.dbg.value(metadata i32 2, metadata !13, metadata !DIExpression())
; CHECK: %inc.2 = add nsw i32 2, 1
; CHECK: call void @llvm.dbg.value(metadata i32 %inc.2, metadata !13, metadata !DIExpression())
; CHECK: ret void

attributes #0 = { nounwind uwtable }
attributes #2 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "dse.c", directory: "C:\5Csrc\5Cllvm-project\5Cbuild")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 6.0.0 "}
!8 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 3, type: !9, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13}
!13 = !DILocalVariable(name: "x", arg: 1, scope: !8, file: !1, line: 3, type: !11)
!14 = !{!15, !15, i64 0}
!15 = !{!"int", !16, i64 0}
!16 = !{!"omnipotent char", !17, i64 0}
!17 = !{!"Simple C/C++ TBAA"}
!18 = !DILocation(line: 3, column: 12, scope: !8)
!19 = !DILocation(line: 4, column: 3, scope: !8)
!20 = !DILocation(line: 5, column: 5, scope: !8)
!21 = !DILocation(line: 6, column: 12, scope: !8)
!22 = !DILocation(line: 6, column: 10, scope: !8)
!23 = !DILocation(line: 7, column: 5, scope: !8)
!24 = !DILocation(line: 8, column: 3, scope: !8)
!25 = !DILocation(line: 9, column: 1, scope: !8)
