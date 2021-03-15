; RUN: opt -S -strip-nonlinetable-debuginfo %s -o - |  FileCheck %s
; int *get() { return new int[256]; }
; ModuleID = '/tmp/heapallocsite.cpp'
source_filename = "/tmp/heapallocsite.cpp"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx11.0.0"

; Function Attrs: noinline optnone ssp uwtable mustprogress
define dso_local i32* @_Z3getv() #0 !dbg !8 {
entry:
; CHECK-LABEL: entry:
; CHECK-NOT: !heapallocsite
  %call = call noalias nonnull i8* @_Znam(i64 1024) #2, !dbg !14, !heapallocsite !13
  %0 = bitcast i8* %call to i32*, !dbg !14
  ret i32* %0, !dbg !15
}

; Function Attrs: nobuiltin allocsize(0)
declare nonnull i8* @_Znam(i64) #1

attributes #0 = { noinline optnone ssp uwtable mustprogress }
attributes #1 = { nobuiltin allocsize(0) "frame-pointer"="all" }
attributes #2 = { builtin allocsize(0) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

; CHECK-LABEL: !0 =
; CHECK-NOT: !DIBasicType(name: "int"
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 13.0.0 (git@github.com:llvm/llvm-project 6d4ce49dae17715de502acbd50ab4c9b3c18215b)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None, sysroot: "/")
!1 = !DIFile(filename: "/tmp/heapallocsite.cpp", directory: "/Volumes/Data/llvm-project")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 13.0.0 (git@github.com:llvm/llvm-project 6d4ce49dae17715de502acbd50ab4c9b3c18215b)"}
!8 = distinct !DISubprogram(name: "get", linkageName: "_Z3getv", scope: !9, file: !9, line: 1, type: !10, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!9 = !DIFile(filename: "/tmp/heapallocsite.cpp", directory: "")
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DILocation(line: 1, column: 21, scope: !8)
!15 = !DILocation(line: 1, column: 14, scope: !8)
