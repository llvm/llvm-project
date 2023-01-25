; RUN: opt -safe-stack -S -mtriple=i386-pc-linux-gnu < %s -o - | FileCheck %s

; Test llvm.dbg.value for dynamic allocas moved onto the unsafe stack.
; In the dynamic alloca case, the dbg.value does not change with the exception
; of the alloca pointer in the first argument being replaced with the new stack
; top address.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @f(i32 %n) safestack !dbg !6 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %n, metadata !11, metadata !14), !dbg !15
  %0 = zext i32 %n to i64, !dbg !16

; CHECK:  store ptr %[[VLA:.*]], ptr @__safestack_unsafe_stack_ptr
; CHECK:  tail call void @llvm.dbg.value(metadata ptr %[[VLA]], metadata ![[TYPE:.*]], metadata !DIExpression(DW_OP_deref))
; CHECK:  call void @capture({{.*}} %[[VLA]])

  %vla = alloca i8, i64 %0, align 16, !dbg !16
  tail call void @llvm.dbg.value(metadata ptr %vla, metadata !12, metadata !17), !dbg !18
  call void @capture(ptr nonnull %vla), !dbg !19
  ret void, !dbg !20
}

declare void @capture(ptr)
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 (trunk 272832) (llvm/trunk 272831)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "../llvm/1.cc", directory: "/code/build-llvm")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 3.9.0 (trunk 272832) (llvm/trunk 272831)"}
!6 = distinct !DISubprogram(name: "f", linkageName: "_Z1fi", scope: !1, file: !1, line: 2, type: !7, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !10)
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9}
!9 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !{!11, !12}
!11 = !DILocalVariable(name: "n", arg: 1, scope: !6, file: !1, line: 2, type: !9)

; CHECK-DAG: ![[TYPE]] = !DILocalVariable(name: "x",
!12 = !DILocalVariable(name: "x", scope: !6, file: !1, line: 3, type: !13)
!13 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!14 = !DIExpression()
!15 = !DILocation(line: 2, column: 12, scope: !6)
!16 = !DILocation(line: 3, column: 3, scope: !6)

!17 = !DIExpression(DW_OP_deref)
!18 = !DILocation(line: 3, column: 8, scope: !6)
!19 = !DILocation(line: 4, column: 3, scope: !6)
!20 = !DILocation(line: 5, column: 1, scope: !6)
