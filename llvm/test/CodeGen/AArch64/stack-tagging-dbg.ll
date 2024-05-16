; RUN: opt < %s -aarch64-stack-tagging -S -o - | FileCheck %s
; RUN: opt < %s -aarch64-stack-tagging -S -o - --try-experimental-debuginfo-iterators | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android"

declare void @use32(ptr)
declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) nounwind readnone speculatable

; Debug intrinsics use the new alloca directly, not through a GEP or a tagp.
define void @DbgIntrinsics() sanitize_memtag {
entry:
  %x = alloca i32, align 4
  call void @llvm.dbg.declare(metadata ptr %x, metadata !6, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata !DIArgList(ptr %x, ptr %x), metadata !6, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus)), !dbg !10
  store i32 42, ptr %x, align 4
  call void @use32(ptr %x)
  ret void
}

; CHECK-LABEL: define void @DbgIntrinsics(
; CHECK:  [[X:%.*]] = alloca { i32, [12 x i8] }, align 16
; CHECK:  call void @llvm.dbg.declare(metadata ptr [[X]],
; CHECK:  call void @llvm.dbg.value(metadata !DIArgList(ptr [[X]], ptr [[X]])


!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 9.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "stack-tagging.cc", directory: "/tmp")
!2 = !{}
!3 = distinct !DISubprogram(name: "DbgIntrinsics", linkageName: "DbgIntrinsics", scope: !1, file: !1, line: 3, type: !4, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!4 = !DISubroutineType(types: !5)
!5 = !{null}
!6 = !DILocalVariable(name: "x", scope: !3, file: !1, line: 4, type: !7)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !DILocation(line: 1, column: 2, scope: !3)
