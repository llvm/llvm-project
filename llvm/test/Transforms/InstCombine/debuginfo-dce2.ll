; RUN: opt -instcombine -S %s -o - | FileCheck %s

; In this example, the cast from i8* to i32* becomes trivially dead. We should
; salvage its debug info.

; C source:
; void use_as_void(void *);
; void f(void *p) {
;   int *q = (int *)p;
;   use_as_void(q);
; }

; ModuleID = '<stdin>'
source_filename = "t.c"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.11.25508"

; Function Attrs: nounwind uwtable
define void @f(i8* %p) !dbg !11 {
entry:
  call void @llvm.dbg.value(metadata i8* %p, metadata !16, metadata !DIExpression()), !dbg !18
  %0 = bitcast i8* %p to i32*, !dbg !19
  call void @llvm.dbg.value(metadata i32* %0, metadata !17, metadata !DIExpression()), !dbg !20
  %1 = bitcast i32* %0 to i8*, !dbg !21
  call void @use_as_void(i8* %1), !dbg !22
  ret void, !dbg !23
}

; CHECK-LABEL: define void @f(i8* %p)
; CHECK: call void @llvm.dbg.value(metadata i8* %p, metadata ![[P_VAR:[0-9]+]], metadata !DIExpression())
; CHECK-NOT: bitcast
; CHECK: call void @llvm.dbg.value(metadata i8* %p, metadata ![[Q_VAR:[0-9]+]], metadata !DIExpression())
; CHECK-NOT: bitcast
; CHECK ret void

; CHECK: ![[P_VAR]] = !DILocalVariable(name: "p", {{.*}})
; CHECK: ![[Q_VAR]] = !DILocalVariable(name: "q", {{.*}})

declare void @use_as_void(i8*)

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7, !8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3)
!1 = !DIFile(filename: "t.c", directory: "C:\5Csrc\5Cllvm-project\5Cbuild", checksumkind: CSK_MD5, checksum: "56c40617ada23a8cccbd9a16bcec57af")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !{i32 2, !"CodeView", i32 1}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 2}
!9 = !{i32 7, !"PIC Level", i32 2}
!10 = !{!"clang version 6.0.0 "}
!11 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 2, type: !12, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !15)
!12 = !DISubroutineType(types: !13)
!13 = !{null, !14}
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!15 = !{!16, !17}
!16 = !DILocalVariable(name: "p", arg: 1, scope: !11, file: !1, line: 2, type: !14)
!17 = !DILocalVariable(name: "q", scope: !11, file: !1, line: 3, type: !4)
!18 = !DILocation(line: 2, column: 14, scope: !11)
!19 = !DILocation(line: 3, column: 12, scope: !11)
!20 = !DILocation(line: 3, column: 8, scope: !11)
!21 = !DILocation(line: 4, column: 15, scope: !11)
!22 = !DILocation(line: 4, column: 3, scope: !11)
!23 = !DILocation(line: 5, column: 1, scope: !11)
