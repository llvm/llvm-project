; A full link keeps both descriptions of f()::x on the selected global.
; RUN: split-file %s %t
; RUN: llvm-link -S %t/a.ll %t/b.ll | FileCheck %s
; RUN: llvm-link --only-needed -S %t/a.ll %t/b.ll | FileCheck %s --check-prefix=NEEDED
; RUN: sed 's/linkonce_odr //' %t/b.ll > %t/strong.ll
; RUN: llvm-link -S %t/a.ll %t/strong.ll | FileCheck %s

; CHECK: @f.x = {{.*}}global i32 7, !dbg ![[X1:[0-9]+]], !dbg ![[X2:[0-9]+]]{{$}}
; CHECK-DAG: ![[X1]] = !DIGlobalVariableExpression(
; CHECK-DAG: ![[X2]] = !DIGlobalVariableExpression(

; --only-needed skips @b_ref, the only use of @f.x in b.ll.
; NEEDED: @f.x = linkonce_odr global i32 7, !dbg !{{[0-9]+}}{{$}}

;--- a.ll
@a_ref = global ptr @f.x
@f.x = linkonce_odr global i32 7, !dbg !0

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!7}
!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !4, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DISubprogram(name: "f", scope: !4, file: !4, type: !6, spFlags: DISPFlagDefinition, unit: !3, retainedNodes: !{!0})
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, runtimeVersion: 0, emissionKind: FullDebug)
!4 = !DIFile(filename: "a.cpp", directory: "/")
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !DISubroutineType(types: !{null})
!7 = !{i32 2, !"Debug Info Version", i32 3}

;--- b.ll
@b_ref = global ptr @f.x
@f.x = linkonce_odr global i32 7, !dbg !0

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!7}
!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !4, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DISubprogram(name: "f", scope: !4, file: !4, type: !6, spFlags: DISPFlagDefinition, unit: !3, retainedNodes: !{!0})
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, runtimeVersion: 0, emissionKind: FullDebug)
!4 = !DIFile(filename: "b.cpp", directory: "/")
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !DISubroutineType(types: !{null})
!7 = !{i32 2, !"Debug Info Version", i32 3}
