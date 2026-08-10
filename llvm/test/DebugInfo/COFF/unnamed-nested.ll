; RUN: llc < %s -filetype=obj -o %t.o
; RUN: llvm-pdbutil dump -types %t.o | FileCheck %s

; C source to regenerate:
; $ clang -g -gcodeview -S -emit-llvm t.c
; $ cat t.c
; struct {
;   union {
;     struct {};
;   };
; } S;

; Test that this compiles without errors.

; CHECK: LF_STRUCTURE{{.*}}<unnamed-tag>::<unnamed-tag>::<unnamed-tag>
; CHECK: LF_UNION{{.*}}<unnamed-tag>::<unnamed-tag>
; CHECK: LF_STRUCTURE{{.*}}<unnamed-tag>

; ModuleID = 't.c'
source_filename = "t.c"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.24.28316"

%struct.anon = type { %union.anon }
%union.anon = type { %struct.anon.0 }
%struct.anon.0 = type { [4 x i8] }

@S = dso_local global %struct.anon zeroinitializer, align 1, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!13, !14, !15, !16}
!llvm.ident = !{!17}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "S", scope: !2, file: !3, line: 5, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 11.0.0 (https://github.com/llvm/llvm-project.git 60d09bec7f8699728d38057430422d955d32a904)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "C:\\src\\llvm-build", checksumkind: CSK_MD5, checksum: "c31fe86676dd2fb56f847f926c0f2c71")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !3, line: 1, size: 32, elements: !7)
!7 = !{!8, !12}
!8 = distinct !DICompositeType(tag: DW_TAG_union_type, scope: !6, file: !3, line: 2, size: 32, elements: !9)
!9 = !{!10, !11}
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, scope: !8, file: !3, line: 3, size: 32, elements: !4)
!11 = !DIDerivedType(tag: DW_TAG_member, scope: !8, file: !3, line: 3, baseType: !10, size: 32)
!12 = !DIDerivedType(tag: DW_TAG_member, scope: !6, file: !3, line: 2, baseType: !8, size: 32)
!13 = !{i32 2, !"CodeView", i32 1}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"wchar_size", i32 2}
!16 = !{i32 7, !"PIC Level", i32 2}
!17 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git 60d09bec7f8699728d38057430422d955d32a904)"}
