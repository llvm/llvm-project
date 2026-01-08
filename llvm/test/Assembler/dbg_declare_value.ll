; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

; CHECK: #dbg_declare_value(double %{{[0-9]+}}, !{{[0-9]+}}, !DIExpression(), !{{[0-9]+}})

; ModuleID = '/tmp/test.c'
source_filename = "/tmp/test.c"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx26.0.0"

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define void @foo(double noundef %0) #0 !dbg !9 {
    #dbg_declare_value(double %0, !15, !DIExpression(), !16)
  ret void, !dbg !21
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 22.0.0git (git@github.com:rastogishubham/llvm-project.git bacf99969b2f3e6db4cfcf536cce8b01ffd20aa0)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
!1 = !DIFile(filename: "/tmp/test.c", directory: "/Users/srastogi/Development/llvm-project-2/build_ninja", checksumkind: CSK_MD5, checksum: "fa15cf45ed4f9d805aab17eb7856a442")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"uwtable", i32 1}
!7 = !{i32 7, !"frame-pointer", i32 4}
!8 = !{!"clang version 22.0.0git (git@github.com:rastogishubham/llvm-project.git bacf99969b2f3e6db4cfcf536cce8b01ffd20aa0)"}
!9 = distinct !DISubprogram(name: "foo", scope: !10, file: !10, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!10 = !DIFile(filename: "/tmp/test.c", directory: "", checksumkind: CSK_MD5, checksum: "fa15cf45ed4f9d805aab17eb7856a442")
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13}
!13 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!14 = !{}
!15 = !DILocalVariable(name: "x", arg: 1, scope: !9, file: !10, line: 1, type: !13)
!16 = !DILocation(line: 1, column: 17, scope: !9)
!21 = !DILocation(line: 3, column: 1, scope: !9)
