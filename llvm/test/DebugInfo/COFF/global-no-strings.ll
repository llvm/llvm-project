; RUN: llc < %s | FileCheck %s
; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s

; This test ensures that no debuginfo is emitted for the constant "123456789"
; string. These global variables have debug expressions because DWARF has the
; ability to tie them to a file name and line number, but this isn't possible
; in CodeView, so we make sure it's omitted to save file size.
;
; The various CodeView outputs should have a description of "my_string", but not
; the string contents itself.

; C++ source to regenerate:
; $ cat a.cpp
; char* my_string =
;   "12345679";
;
; $ clang-cl a.cpp /GS- /Z7 /GR- /clang:-S /clang:-emit-llvm

; CHECK-NOT: S_LDATA
; CHECK: S_GDATA
; CHECK-NOT: S_LDATA
; CHECK-NOT: S_GDATA

; ModuleID = '/tmp/a.c'
source_filename = "/tmp/a.c"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.20.0"

$"??_C@_08HCBFHPJA@12345679?$AA@" = comdat any

@"??_C@_08HCBFHPJA@12345679?$AA@" = linkonce_odr dso_local unnamed_addr constant [9 x i8] c"12345679\00", comdat, align 1, !dbg !0
@my_string = dso_local global ptr @"??_C@_08HCBFHPJA@12345679?$AA@", align 8, !dbg !7

!llvm.dbg.cu = !{!9}
!llvm.linker.options = !{!13, !14}
!llvm.module.flags = !{!15, !16, !17, !18, !19}
!llvm.ident = !{!20}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(scope: null, file: !2, line: 2, type: !3, isLocal: true, isDefinition: true)
!2 = !DIFile(filename: "/tmp/a.c", directory: "", checksumkind: CSK_MD5, checksum: "b972961d64de3c90497767110ab58eb6")
!3 = !DICompositeType(tag: DW_TAG_array_type, baseType: !4, size: 72, elements: !5)
!4 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!5 = !{!6}
!6 = !DISubrange(count: 9)
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = distinct !DIGlobalVariable(name: "my_string", scope: !9, file: !2, line: 1, type: !12, isLocal: false, isDefinition: true)
!9 = distinct !DICompileUnit(language: DW_LANG_C99, file: !10, producer: "clang version 15.0.0 (https://github.com/llvm/llvm-project.git 7c1c0849f8a1a6f1bf5f5b554484bbf8b0debd0a)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !11, splitDebugInlining: false, nameTableKind: None)
!10 = !DIFile(filename: "/tmp/a.c", directory: "/usr/local/google/home/mitchp/llvm-build/opt", checksumkind: CSK_MD5, checksum: "b972961d64de3c90497767110ab58eb6")
!11 = !{!0, !7}
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !4, size: 64)
!13 = !{!"/DEFAULTLIB:libcmt.lib"}
!14 = !{!"/DEFAULTLIB:oldnames.lib"}
!15 = !{i32 2, !"CodeView", i32 1}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = !{i32 1, !"wchar_size", i32 2}
!18 = !{i32 7, !"PIC Level", i32 2}
!19 = !{i32 7, !"uwtable", i32 2}
!20 = !{!"clang version 15.0.0 (https://github.com/llvm/llvm-project.git 7c1c0849f8a1a6f1bf5f5b554484bbf8b0debd0a)"}
