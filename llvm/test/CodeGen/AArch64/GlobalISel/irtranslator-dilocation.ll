; RUN: llc -O0 -mtriple=aarch64-apple-ios -global-isel -debug-only=irtranslator \
; RUN:     -stop-after=irtranslator %s -o - 2>&1 | FileCheck %s
; RUN: llc -O0 -mtriple=aarch64-apple-ios -global-isel -debug-only=irtranslator \
; RUN:     -stop-after=irtranslator %s -o - 2>&1 --try-experimental-debuginfo-iterators | FileCheck %s

; REQUIRES: asserts

; CHECK: Checking DILocation from   %retval = alloca i32, align 4{{.*}} was copied to G_FRAME_INDEX
; CHECK: Checking DILocation from   %rv = alloca i32, align 4{{.*}} was copied to G_FRAME_INDEX
; CHECK: Checking DILocation from   store i32 0, ptr %retval, align 4{{.*}} was copied to G_CONSTANT
; CHECK: Checking DILocation from   store i32 0, ptr %retval, align 4{{.*}} was copied to G_STORE
; CHECK: Checking DILocation from   store i32 0, ptr %rv, align 4, !dbg !12{{.*}} was copied to G_STORE debug-location !12; t.cpp:2:5
; CHECK: Checking DILocation from   %0 = load i32, ptr %rv, align 4, !dbg !13{{.*}} was copied to G_LOAD debug-location !13; t.cpp:3:8
; CHECK: Checking DILocation from   ret i32 %0, !dbg !14{{.*}} was copied to COPY debug-location !14; t.cpp:3:1
; CHECK: Checking DILocation from   ret i32 %0, !dbg !14{{.*}} was copied to RET_ReallyLR implicit $w0, debug-location !14; t.cpp:3:1

source_filename = "t.cpp"
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "arm64-unknown-linux-gnu"

; Function Attrs: noinline norecurse nounwind optnone
define dso_local i32 @main() !dbg !7 {
entry:
  %retval = alloca i32, align 4
  %rv = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  call void @llvm.dbg.declare(metadata ptr %rv, metadata !11, metadata !DIExpression()), !dbg !12
  store i32 0, ptr %rv, align 4, !dbg !12
  %0 = load i32, ptr %rv, align 4, !dbg !13
  ret i32 %0, !dbg !14
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 8.0.0 (trunk) (llvm/trunk 344296)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "t.cpp", directory: "/Volumes/Data/llvm.org/svn/build")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 8.0.0 (trunk) (llvm/trunk 344296)"}
!7 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocalVariable(name: "rv", scope: !7, file: !1, line: 2, type: !10)
!12 = !DILocation(line: 2, column: 5, scope: !7)
!13 = !DILocation(line: 3, column: 8, scope: !7)
!14 = !DILocation(line: 3, column: 1, scope: !7)

