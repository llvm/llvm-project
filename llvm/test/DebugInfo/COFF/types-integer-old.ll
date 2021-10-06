; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s

; Tests that CodeView integer types are generated even when using Clang's old integer type names.

; C++ source to regenerate:
; $ cat t.cpp
; void usevars(long, ...);
; void f() {
;   long l1 = 0;
;   unsigned long l2 = 0;
;   usevars(l1, l2);
; }
; $ clang t.cpp -S -emit-llvm -g -gcodeview -o t.ll  -target x86_64-pc-windows-msvc19.0.23918

; CHECK:     LocalSym {
; CHECK:       Type: long (0x12)
; CHECK:       VarName: l1
; CHECK:     }
; CHECK:     LocalSym {
; CHECK:       Type: unsigned long (0x22)
; CHECK:       VarName: l2
; CHECK:     }

; ModuleID = '/usr/local/google/home/blaikie/dev/scratch/t.cpp'
source_filename = "/usr/local/google/home/blaikie/dev/scratch/t.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.23918"

; Function Attrs: mustprogress noinline optnone uwtable
define dso_local void @"?f@@YAXXZ"() #0 !dbg !8 {
entry:
  %l1 = alloca i32, align 4
  %l2 = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %l1, metadata !13, metadata !DIExpression()), !dbg !15
  store i32 0, i32* %l1, align 4, !dbg !15
  call void @llvm.dbg.declare(metadata i32* %l2, metadata !16, metadata !DIExpression()), !dbg !18
  store i32 0, i32* %l2, align 4, !dbg !18
  %0 = load i32, i32* %l2, align 4, !dbg !19
  %1 = load i32, i32* %l1, align 4, !dbg !19
  call void (i32, ...) @"?usevars@@YAXJZZ"(i32 %1, i32 %0), !dbg !19
  ret void, !dbg !20
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare dso_local void @"?usevars@@YAXJZZ"(i32, ...) #2

attributes #0 = { mustprogress noinline optnone uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 14.0.0 (git@github.com:llvm/llvm-project.git 3709fb72c86bea1f0e6c51ab334ed6417cbe1c07)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/usr/local/google/home/blaikie/dev/scratch/t.cpp", directory: "/usr/local/google/home/blaikie/dev/llvm/src", checksumkind: CSK_MD5, checksum: "a8e7ccc989ea91d67d3cb95afa046aa5")
!2 = !{i32 2, !"CodeView", i32 1}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 2}
!5 = !{i32 7, !"PIC Level", i32 2}
!6 = !{i32 7, !"uwtable", i32 1}
!7 = !{!"clang version 14.0.0 (git@github.com:llvm/llvm-project.git 3709fb72c86bea1f0e6c51ab334ed6417cbe1c07)"}
!8 = distinct !DISubprogram(name: "f", linkageName: "?f@@YAXXZ", scope: !9, file: !9, line: 2, type: !10, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
!9 = !DIFile(filename: "scratch/t.cpp", directory: "/usr/local/google/home/blaikie/dev", checksumkind: CSK_MD5, checksum: "a8e7ccc989ea91d67d3cb95afa046aa5")
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !{}
!13 = !DILocalVariable(name: "l1", scope: !8, file: !9, line: 3, type: !14)
!14 = !DIBasicType(name: "long int", size: 32, encoding: DW_ATE_signed)
!15 = !DILocation(line: 3, scope: !8)
!16 = !DILocalVariable(name: "l2", scope: !8, file: !9, line: 4, type: !17)
!17 = !DIBasicType(name: "long unsigned int", size: 32, encoding: DW_ATE_unsigned)
!18 = !DILocation(line: 4, scope: !8)
!19 = !DILocation(line: 5, scope: !8)
!20 = !DILocation(line: 6, scope: !8)
