; RUN: opt -S -loop-reduce %s | FileCheck %s
;
; Test that LSR SCEV-based salvaging does not crash when translating SCEVs
; that contain integers with binary representations greater than 64-bits. 
; Also show that no salvaging attempt is made for dbg.value that are undef
; pre-LSR.
;
; CHECK: call void @llvm.dbg.value(metadata i64 undef, metadata !{{[0-9]+}}, metadata !DIExpression(DW_OP_plus_uconst, 228, DW_OP_stack_value))
; CHECK: call void @llvm.dbg.value(metadata i64 %var2, metadata !{{[0-9]+}}, metadata !DIExpression(DW_OP_plus_uconst, 228, DW_OP_stack_value))


target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

; Function Attrs: nounwind
define hidden void @reproducer() local_unnamed_addr !dbg !5 {
init:
  %0 = lshr i128 undef, 64
  %var1 = trunc i128 %0 to i64
  %1 = add nuw i64 undef, %var1
  %var2 = lshr i64 %1, 12
  br label %Label_d0

Label_d0:                                         ; preds = %Label_d0, %init
  %var3 = phi i64 [ %var2, %init ], [ %var4, %Label_d0 ]
  call void @llvm.dbg.value(metadata i64 undef, metadata !11, metadata !DIExpression(DW_OP_plus_uconst, 228, DW_OP_stack_value)), !dbg !12
  call void @llvm.dbg.value(metadata i64 %var2, metadata !11, metadata !DIExpression(DW_OP_plus_uconst, 228, DW_OP_stack_value)), !dbg !12
  %var4 = add i64 %var3, -1
  %var5 = icmp eq i64 %var4, 0
  br i1 %var5, label %Label_1bc, label %Label_d0

Label_1bc:                                        ; preds = %Label_d0
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "frontend", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "source", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = distinct !DISubprogram(name: "reproducer", scope: !1, file: !1, line: 904320, type: !6, scopeLine: 904320, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !10)
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8, !9, !9, !9, !9, !9, !9}
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!9 = !DIBasicType(name: "my_type", size: 64, encoding: DW_ATE_unsigned)
!10 = !{!11}
!11 = !DILocalVariable(name: "my_var", arg: 1, scope: !5, file: !1, line: 904320, type: !8)
!12 = !DILocation(line: 904544, scope: !5)