; RUN: opt < %s -mem2reg -S | FileCheck %s

define double @testfunc(i32 %i, double %j) nounwind ssp !dbg !1 {
entry:
  %i_addr = alloca i32                            ; <i32*> [#uses=2]
  %j_addr = alloca double                         ; <double*> [#uses=2]
  %retval = alloca double                         ; <double*> [#uses=2]
  %0 = alloca double                              ; <double*> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata i32* %i_addr, metadata !0, metadata !DIExpression()), !dbg !8
; CHECK: call void @llvm.dbg.value(metadata i32 %i, metadata ![[IVAR:[0-9]*]], metadata {{.*}})
; CHECK: call void @llvm.dbg.value(metadata double %j, metadata ![[JVAR:[0-9]*]], metadata {{.*}})
; CHECK: ![[IVAR]] = !DILocalVariable(name: "i"
; CHECK: ![[JVAR]] = !DILocalVariable(name: "j"
  store i32 %i, i32* %i_addr
  call void @llvm.dbg.declare(metadata double* %j_addr, metadata !9, metadata !DIExpression()), !dbg !8
  store double %j, double* %j_addr
  %1 = load i32, i32* %i_addr, align 4, !dbg !10       ; <i32> [#uses=1]
  %2 = add nsw i32 %1, 1, !dbg !10                ; <i32> [#uses=1]
  %3 = sitofp i32 %2 to double, !dbg !10          ; <double> [#uses=1]
  %4 = load double, double* %j_addr, align 8, !dbg !10    ; <double> [#uses=1]
  %5 = fadd double %3, %4, !dbg !10               ; <double> [#uses=1]
  store double %5, double* %0, align 8, !dbg !10
  %6 = load double, double* %0, align 8, !dbg !10         ; <double> [#uses=1]
  store double %6, double* %retval, align 8, !dbg !10
  br label %return, !dbg !10

return:                                           ; preds = %entry
  %retval1 = load double, double* %retval, !dbg !10       ; <double> [#uses=1]
  ret double %retval1, !dbg !10
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!14}

!0 = !DILocalVariable(name: "i", line: 2, arg: 1, scope: !1, file: !2, type: !7)
!1 = distinct !DISubprogram(name: "testfunc", linkageName: "testfunc", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !3, scopeLine: 2, file: !12, scope: !2, type: !4)
!2 = !DIFile(filename: "testfunc.c", directory: "/tmp")
!3 = distinct !DICompileUnit(language: DW_LANG_C89, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: true, emissionKind: FullDebug, file: !12, enums: !13, retainedTypes: !13)
!4 = !DISubroutineType(types: !5)
!5 = !{!6, !7, !6}
!6 = !DIBasicType(tag: DW_TAG_base_type, name: "double", size: 64, align: 64, encoding: DW_ATE_float)
!7 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !DILocation(line: 2, scope: !1)
!9 = !DILocalVariable(name: "j", line: 2, arg: 2, scope: !1, file: !2, type: !6)
!10 = !DILocation(line: 3, scope: !11)
!11 = distinct !DILexicalBlock(line: 2, column: 0, file: !12, scope: !1)
!12 = !DIFile(filename: "testfunc.c", directory: "/tmp")
!13 = !{}
!14 = !{i32 1, !"Debug Info Version", i32 3}
