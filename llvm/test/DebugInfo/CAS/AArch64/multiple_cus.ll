; RUN: llc -cas-friendly-debug-info -O0 --filetype=obj --cas-backend --cas=%t/cas --mccas-verify --mtriple=arm64-apple-darwin %s -o %t/multiple_cus.id 2>&1 | FileCheck %s --allow-empty
; CHECK-NOT: error in backend: CASBackend output round-trip verification error

; ModuleID = 'llvm-link'
source_filename = "llvm-link"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx14.0.0"

; Function Attrs: mustprogress noinline nounwind optnone ssp uwtable(sync)
define noundef i32 @_Z4foo2i(i32 noundef %a) #0 !dbg !11 {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %a.addr, metadata !17, metadata !DIExpression()), !dbg !18
  %0 = load i32, ptr %a.addr, align 4, !dbg !18
  %add = add nsw i32 %0, 5, !dbg !18
  ret i32 %add, !dbg !18
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: mustprogress noinline nounwind optnone ssp uwtable(sync)
define noundef i32 @_Z3fooi(i32 noundef %x) #0 !dbg !22 {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %x.addr, metadata !23, metadata !DIExpression()), !dbg !24
  %0 = load i32, ptr %x.addr, align 4, !dbg !24
  %add = add nsw i32 %0, 2, !dbg !24
  ret i32 %add, !dbg !24
}

; Function Attrs: mustprogress noinline nounwind optnone ssp uwtable(sync)
define noundef float @_Z3barf(float noundef %a) #0 !dbg !28 {
entry:
  %a.addr = alloca float, align 4
  store float %a, ptr %a.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %a.addr, metadata !33, metadata !DIExpression()), !dbg !34
  %0 = load float, ptr %a.addr, align 4, !dbg !34
  %mul = fmul float %0, 5.000000e+00, !dbg !34
  ret float %mul, !dbg !34
}

; Function Attrs: mustprogress noinline nounwind optnone ssp uwtable(sync)
define noundef float @_Z4bar2f(float noundef %x) #0 !dbg !38 {
entry:
  %x.addr = alloca float, align 4
  store float %x, ptr %x.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %x.addr, metadata !39, metadata !DIExpression()), !dbg !40
  %0 = load float, ptr %x.addr, align 4, !dbg !40
  %mul = fmul float %0, 2.000000e+00, !dbg !40
  ret float %mul, !dbg !40
}

!llvm.dbg.cu = !{!0, !2}
!llvm.ident = !{!4, !4}
!llvm.module.flags = !{!5, !6, !7, !8, !9, !10}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 18.0.0 (git@github.com:apple/llvm-project.git c9c19151210716562c81949ac2e5174e58bc8b64)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
!1 = !DIFile(filename: "/Users/shubham/Development/test109275485/a.cpp", directory: "/Users/shubham/Development/llvm-project-cas/llvm-project")
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 18.0.0 (git@github.com:apple/llvm-project.git c9c19151210716562c81949ac2e5174e58bc8b64)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
!3 = !DIFile(filename: "/Users/shubham/Development/test109275485/b.cpp", directory: "/Users/shubham/Development/llvm-project-cas/llvm-project")
!4 = !{!"clang version 18.0.0 (git@github.com:apple/llvm-project.git)"}
!5 = !{i32 7, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{i32 8, !"PIC Level", i32 2}
!9 = !{i32 7, !"uwtable", i32 1}
!10 = !{i32 7, !"frame-pointer", i32 1}
!11 = distinct !DISubprogram(name: "foo2", linkageName: "_Z4foo2i", scope: !12, file: !12, line: 2, type: !13, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !16)
!12 = !DIFile(filename: "test109275485/a.cpp", directory: "/Users/shubham/Development")
!13 = !DISubroutineType(types: !14)
!14 = !{!15, !15}
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !{}
!17 = !DILocalVariable(name: "a", arg: 1, scope: !11, file: !12, line: 2, type: !15)
!18 = !DILocation(line: 2, column: 14, scope: !11)
!22 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooi", scope: !12, file: !12, line: 6, type: !13, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !16)
!23 = !DILocalVariable(name: "x", arg: 1, scope: !22, file: !12, line: 6, type: !15)
!24 = !DILocation(line: 6, column: 13, scope: !22)
!28 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barf", scope: !29, file: !29, line: 1, type: !30, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !16)
!29 = !DIFile(filename: "test109275485/b.cpp", directory: "/Users/shubham/Development")
!30 = !DISubroutineType(types: !31)
!31 = !{!32, !32}
!32 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!33 = !DILocalVariable(name: "a", arg: 1, scope: !28, file: !29, line: 1, type: !32)
!34 = !DILocation(line: 1, column: 17, scope: !28)
!38 = distinct !DISubprogram(name: "bar2", linkageName: "_Z4bar2f", scope: !29, file: !29, line: 5, type: !30, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !16)
!39 = !DILocalVariable(name: "x", arg: 1, scope: !38, file: !29, line: 5, type: !32)
!40 = !DILocation(line: 5, column: 18, scope: !38)

