; RUN: opt -S -passes=globalopt < %s | FileCheck %s

;; Generated at -g from:
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;subroutine sub(inode,node)
;;  implicit none
;;  integer nodea,nodeb,jmax,inode,node
;;  save nodea,nodeb,jmax
;;  if(inode.eq.1) then
;;    nodea=node
;;    nodeb=node
;;    jmax=0
;;  else
;;    jmax=node+1+nodea+nodeb+jmax
;;    return
;;  endif
;;end
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

%struct.BSS1 = type <{ [12 x i8] }>

;CHECK: @.BSS1.0 = internal unnamed_addr global i32 0, align 32, !dbg ![[GVE1:.*]]
;CHECK: @.BSS1.1 = internal unnamed_addr global i32 0, align 4, !dbg ![[GVE2:.*]], !dbg ![[GVE4:.*]]
;CHECK: @.BSS1.2 = internal unnamed_addr global i32 0, align 8, !dbg ![[GVE3:.*]]

@.BSS1 = internal global %struct.BSS1 zeroinitializer, align 32, !dbg !0, !dbg !7, !dbg !10, !dbg !27, !dbg !29
@.C330_sub_ = internal constant i32 0
@.C332_sub_ = internal constant i32 1

define void @sub_(ptr noalias %inode, ptr noalias %node) !dbg !2 {
L.entry:
  call void @llvm.dbg.declare(metadata ptr %inode, metadata !16, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.declare(metadata ptr %node, metadata !18, metadata !DIExpression()), !dbg !17
  br label %L.LB1_360

L.LB1_360:                                        ; preds = %L.entry
  %0 = load i32, ptr %inode, align 4, !dbg !19
  %1 = icmp ne i32 %0, 1, !dbg !19
  br i1 %1, label %L.LB1_356, label %L.LB1_370, !dbg !19

L.LB1_370:                                        ; preds = %L.LB1_360
  %2 = load i32, ptr %node, align 4, !dbg !20
  store i32 %2, ptr @.BSS1, align 4, !dbg !20
  %3 = load i32, ptr %node, align 4, !dbg !21
  %4 = getelementptr i8, ptr @.BSS1, i64 4, !dbg !21
  store i32 %3, ptr %4, align 4, !dbg !21
  %5 = getelementptr i8, ptr @.BSS1, i64 8, !dbg !22
  store i32 0, ptr %5, align 4, !dbg !22
  br label %L.LB1_357, !dbg !23

L.LB1_356:                                        ; preds = %L.LB1_360
  %6 = load i32, ptr %node, align 4, !dbg !24
  %7 = add nsw i32 %6, 1, !dbg !24
  %8 = load i32, ptr @.BSS1, align 4, !dbg !24
  %9 = add nsw i32 %7, %8, !dbg !24
  %10 = getelementptr i8, ptr @.BSS1, i64 4, !dbg !24
  %11 = load i32, ptr %10, align 4, !dbg !24
  %12 = add nsw i32 %9, %11, !dbg !24
  %13 = getelementptr i8, ptr @.BSS1, i64 8, !dbg !24
  %14 = load i32, ptr %13, align 4, !dbg !24
  %15 = add nsw i32 %12, %14, !dbg !24
  %16 = getelementptr i8, ptr @.BSS1, i64 8, !dbg !24
  store i32 %15, ptr %16, align 4, !dbg !24
  br label %L.LB1_354, !dbg !25

L.LB1_357:                                        ; preds = %L.LB1_370
  br label %L.LB1_354

L.LB1_354:                                        ; preds = %L.LB1_357, %L.LB1_356
  ret void, !dbg !26
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.module.flags = !{!14, !15}
!llvm.dbg.cu = !{!4}

; CHECK-DAG: ![[GVE1]] = !DIGlobalVariableExpression(var: ![[GV1:.*]], expr: !DIExpression())
; CHECK-DAG: ![[GV1]] = distinct !DIGlobalVariable(name: "nodea"
; CHECK-DAG: ![[GVE2]] = !DIGlobalVariableExpression(var: ![[GV2:.*]], expr: !DIExpression())
; CHECK-DAG: ![[GV2]] = distinct !DIGlobalVariable(name: "nodeb"
; CHECK-DAG: ![[GVE3]] = !DIGlobalVariableExpression(var: ![[GV3:.*]], expr: !DIExpression())
; CHECK-DAG: ![[GV3]] = distinct !DIGlobalVariable(name: "jmax"
; CHECK-DAG: ![[GVE4]] = !DIGlobalVariableExpression(var: ![[GV4:.*]], expr: !DIExpression(DW_OP_plus_uconst, 2))
; CHECK-DAG: ![[GV4]] = distinct !DIGlobalVariable(name: "ivar2"

;; This does not make sense should be removed.
; CHECK-NOT: !DIGlobalVariable(name: "doesnotexist"

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "nodea", scope: !2, file: !3, line: 3, type: !9, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "sub", scope: !4, file: !3, line: 1, type: !12, scopeLine: 1, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition, unit: !4)
!3 = !DIFile(filename: "global-sra-struct-fit-segment.f90", directory: "/tmp")
!4 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, flags: "'+flang -g -O0 -S -emit-llvm'", runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6, imports: !5, nameTableKind: None)
!5 = !{}
!6 = !{!0, !7, !10, !29}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression(DW_OP_plus_uconst, 4))
!8 = distinct !DIGlobalVariable(name: "nodeb", scope: !2, file: !3, line: 3, type: !9, isLocal: true, isDefinition: true)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression(DW_OP_plus_uconst, 8))
!11 = distinct !DIGlobalVariable(name: "jmax", scope: !2, file: !3, line: 3, type: !9, isLocal: true, isDefinition: true)
!12 = !DISubroutineType(types: !13)
!13 = !{null, !9, !9}
!14 = !{i32 2, !"Dwarf Version", i32 4}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !DILocalVariable(name: "inode", arg: 1, scope: !2, file: !3, line: 1, type: !9)
!17 = !DILocation(line: 0, scope: !2)
!18 = !DILocalVariable(name: "node", arg: 2, scope: !2, file: !3, line: 1, type: !9)
!19 = !DILocation(line: 5, column: 1, scope: !2)
!20 = !DILocation(line: 6, column: 1, scope: !2)
!21 = !DILocation(line: 7, column: 1, scope: !2)
!22 = !DILocation(line: 8, column: 1, scope: !2)
!23 = !DILocation(line: 9, column: 1, scope: !2)
!24 = !DILocation(line: 10, column: 1, scope: !2)
!25 = !DILocation(line: 11, column: 1, scope: !2)
!26 = !DILocation(line: 13, column: 1, scope: !2)
!27 = !DIGlobalVariableExpression(var: !28, expr: !DIExpression(DW_OP_constu, 4, DW_OP_minus))
!28 = distinct !DIGlobalVariable(name: "doesnotexist", scope: !2, file: !3, line: 3, type: !9, isLocal: true, isDefinition: true)
!29 = !DIGlobalVariableExpression(var: !30, expr: !DIExpression(DW_OP_plus_uconst, 6))
!30 = distinct !DIGlobalVariable(name: "ivar2", scope: !2, file: !3, line: 3, type: !31, isLocal: true, isDefinition: true)
!31 = !DIBasicType(name: "integer*2", size: 16, align: 16, encoding: DW_ATE_signed)
