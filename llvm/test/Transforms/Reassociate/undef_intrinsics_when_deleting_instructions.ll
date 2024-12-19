; RUN: opt < %s -passes=reassociate -S | FileCheck %s
; RUN: opt < %s -passes=reassociate -S --try-experimental-debuginfo-iterators | FileCheck %s

; Check that reassociate pass now undefs debug intrinsics that reference a value
; that gets dropped and cannot be salvaged.

; CHECK-NOT: %add = fadd fast float %a, %b
; CHECK: #dbg_value(float poison, [[VAR_X:![0-9]+]], !DIExpression(),

; CHECK-LABEL: if.then:
; CHECK-NOT: %add1 = fadd fast float %add, %c
; CHECK: #dbg_value(float poison, [[VAR_Y:![0-9]+]], !DIExpression(),
; CHECK-LABEL: !0 =
; CHECK-DAG: [[VAR_Y]] = !DILocalVariable(name: "y"
; CHECK-DAG: [[VAR_X]] = !DILocalVariable(name: "x"

define float @"?foo@@YAMMMMM@Z"(float %a, float %b, float %c, float %d) !dbg !8 {
entry:
  call void @llvm.dbg.value(metadata float %d, metadata !12, metadata !DIExpression()), !dbg !13
  call void @llvm.dbg.value(metadata float %c, metadata !14, metadata !DIExpression()), !dbg !13
  call void @llvm.dbg.value(metadata float %b, metadata !15, metadata !DIExpression()), !dbg !13
  call void @llvm.dbg.value(metadata float %a, metadata !16, metadata !DIExpression()), !dbg !13
  %add = fadd fast float %a, %b, !dbg !17
  call void @llvm.dbg.value(metadata float %add, metadata !18, metadata !DIExpression()), !dbg !13
  %cmp = fcmp fast oeq float %d, 4.000000e+00, !dbg !19
  br i1 %cmp, label %if.then, label %return, !dbg !19

if.then:                                          ; preds = %entry
  %add1 = fadd fast float %add, %c, !dbg !20
  call void @llvm.dbg.value(metadata float %add1, metadata !23, metadata !DIExpression()), !dbg !24
  %sub = fsub fast float %add, 1.200000e+01, !dbg !25
  %sub2 = fsub fast float %add1, %sub, !dbg !25
  %mul = fmul fast float %sub2, 2.000000e+01, !dbg !25
  %div = fdiv fast float %mul, 3.000000e+00, !dbg !25
  br label %return, !dbg !25

return:                                           ; preds = %entry, %if.then
  %retval.0 = phi float [ %div, %if.then ], [ 0.000000e+00, %entry ], !dbg !13
  ret float %retval.0, !dbg !26
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 11.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "undef_intrinsics_when_deleting_instructions.cpp", directory: "/")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 11.0.0"}
!8 = distinct !DISubprogram(name: "foo", linkageName: "?foo@@YAMMMMM@Z", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11, !11, !11, !11}
!11 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!12 = !DILocalVariable(name: "d", arg: 4, scope: !8, file: !1, line: 1, type: !11)
!13 = !DILocation(line: 0, scope: !8)
!14 = !DILocalVariable(name: "c", arg: 3, scope: !8, file: !1, line: 1, type: !11)
!15 = !DILocalVariable(name: "b", arg: 2, scope: !8, file: !1, line: 1, type: !11)
!16 = !DILocalVariable(name: "a", arg: 1, scope: !8, file: !1, line: 1, type: !11)
!17 = !DILocation(line: 2, scope: !8)
!18 = !DILocalVariable(name: "x", scope: !8, file: !1, line: 2, type: !11)
!19 = !DILocation(line: 3, scope: !8)
!20 = !DILocation(line: 4, scope: !21)
!21 = distinct !DILexicalBlock(scope: !22, file: !1, line: 3)
!22 = distinct !DILexicalBlock(scope: !8, file: !1, line: 3)
!23 = !DILocalVariable(name: "y", scope: !21, file: !1, line: 4, type: !11)
!24 = !DILocation(line: 0, scope: !21)
!25 = !DILocation(line: 5, scope: !21)
!26 = !DILocation(line: 8, scope: !8)
