; RUN: opt -S -strip-debug < %s | FileCheck %s
; RUN: opt -S -strip-debug < %s.bc | FileCheck %s

; Confirm we don't leave around any of the heterogeneous debug-info when
; explicitly stripping debug info via e.g. opt -strip-debug

; Allows for bitcode/IR from amd-staging with heterogeneous-debug metadata to
; be stripped to a form which upstream main can ingest, for example:
; $ cat llvm/test/DebugInfo/heterogeneous-strip-debug.ll | \
;   ~/llvm-project/amd-staging/build/opt -S -strip-debug | \
;   ~/llvm-project/main/build/opt ...

; Note the blanket CHECK-NOT patterns cannot include simply "dbg.def"
; or "dbg.kill" because -strip-debug does not bother ensuring unused debug info
; intrinsic declarations are cleaned up.

; CHECK-NOT: !dbg.def
; CHECK-NOT: call void @llvm.dbg.def
; CHECK-NOT: call void @llvm.dbg.kill
; CHECK-NOT: !llvm.dbg.retainedNodes
; CHECK-NOT: !DIExpr(
; CHECK-NOT: !DILifetime(
; CHECK-NOT: DIOp{{[A-Za-z]+}}(

; Originally generated via:
; echo 'int g = 42; void f() { int l = 42; }' | clang -x c - -o - -O0 -gheterogeneous-dwarf=diexpr -emit-llvm -S

source_filename = "-"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@g = dso_local global i32 42, align 4, !dbg.def !0

define dso_local void @f() #0 !dbg !15 {
entry:
  %l = alloca i32, align 4
  call void @llvm.dbg.def(metadata !20, metadata ptr %l), !dbg !21
  store i32 42, ptr %l, align 4, !dbg !21
  call void @llvm.dbg.kill(metadata !20), !dbg !21
  ret void, !dbg !22
}

declare void @llvm.dbg.def(metadata, metadata) #1
declare void @llvm.dbg.kill(metadata) #1

attributes #0 = { noinline nounwind optnone uwtable }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!1}
!llvm.dbg.retainedNodes = !{!3}
!llvm.module.flags = !{!7, !8, !9, !10, !11, !12, !13}
!llvm.ident = !{!14}

!0 = distinct !DIFragment()
!1 = distinct !DICompileUnit(language: DW_LANG_C11, file: !2, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!2 = !DIFile(filename: "-", directory: "/")
!3 = distinct !DILifetime(object: !4, location: !DIExpr(DIOpArg(0, ptr), DIOpDeref(i32)), argObjects: {!0})
!4 = distinct !DIGlobalVariable(name: "g", scope: !1, file: !5, line: 1, type: !6, isLocal: false, isDefinition: true)
!5 = !DIFile(filename: "<stdin>", directory: "/")
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 7, !"Dwarf Version", i32 5}
!8 = !{i32 4, !"Debug Info Version", i32 4}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{i32 8, !"PIC Level", i32 2}
!11 = !{i32 7, !"PIE Level", i32 2}
!12 = !{i32 7, !"uwtable", i32 2}
!13 = !{i32 7, !"frame-pointer", i32 2}
!14 = !{!"clang"}
!15 = distinct !DISubprogram(name: "f", scope: !5, file: !5, line: 1, type: !16, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !18)
!16 = !DISubroutineType(types: !17)
!17 = !{null}
!18 = !{!19}
!19 = !DILocalVariable(name: "l", scope: !15, file: !5, line: 1, type: !6)
!20 = distinct !DILifetime(object: !19, location: !DIExpr(DIOpReferrer(ptr), DIOpDeref(i32)))
!21 = !DILocation(line: 1, column: 28, scope: !15)
!22 = !DILocation(line: 1, column: 36, scope: !15)
