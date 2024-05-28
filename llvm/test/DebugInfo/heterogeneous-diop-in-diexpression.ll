; RUN: opt -S -passes=verify < %s | FileCheck %s

; FIXME: Test minimal parsing and printing in IR; can replace test when bitcode
; (de)serialization is supported.

; ModuleID = '<stdin>'
source_filename = "<stdin>"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@glob = global i32 42, align 4, !dbg !0

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @func() #0 !dbg !13 {
entry:
  %var = alloca i32, align 4
  ; CHECK: #dbg_value(!DIArgList(ptr %var), !18, !DIExpression(DIOpArg(0, ptr), DIOpFragment(1, 2), DIOpDeref(i32)),
  tail call void @llvm.dbg.value(metadata !DIArgList(ptr %var), metadata !18, metadata !DIExpression(DIOpArg(0, ptr), DIOpFragment(1, 2), DIOpDeref(i32))), !dbg !19
  ret void, !dbg !20
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "target-cpu"="x86-64" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!5, !6, !7, !8, !9, !10, !11}
!llvm.ident = !{!12}

; CHECK: !DIGlobalVariableExpression(var: !1, expr: !DIExpression(DIOpArg(0, ptr), DIOpDeref(i32)))
!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression(DIOpArg(0, ptr), DIOpDeref(i32)))
!1 = distinct !DIGlobalVariable(name: "glob", scope: !2, file: !3, line: 1, type: !4, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 19.0.0git (git@github.com:slinder1/llvm-project.git e4263955383c3e364bd752d02fc44cf5f22143ef)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "-", directory: "/home/slinder1/llvm-project/main", checksumkind: CSK_MD5, checksum: "9e51994790e4105fa7153a61c95a824f")
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = !{i32 7, !"Dwarf Version", i32 5}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{i32 8, !"PIC Level", i32 2}
!9 = !{i32 7, !"PIE Level", i32 2}
!10 = !{i32 7, !"uwtable", i32 2}
!11 = !{i32 7, !"frame-pointer", i32 2}
!12 = !{!"clang version 19.0.0git (git@github.com:slinder1/llvm-project.git e4263955383c3e364bd752d02fc44cf5f22143ef)"}
!13 = distinct !DISubprogram(name: "func", scope: !14, file: !14, line: 15, type: !15, scopeLine: 15, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !17)
!14 = !DIFile(filename: "<stdin>", directory: "/home/slinder1/llvm-project/main", checksumkind: CSK_MD5, checksum: "9e51994790e4105fa7153a61c95a824f")
!15 = !DISubroutineType(types: !16)
!16 = !{null}
!17 = !{}
!18 = !DILocalVariable(name: "var", scope: !13, file: !14, line: 16, type: !4)
!19 = !DILocation(line: 16, column: 9, scope: !13)
!20 = !DILocation(line: 17, column: 1, scope: !13)
