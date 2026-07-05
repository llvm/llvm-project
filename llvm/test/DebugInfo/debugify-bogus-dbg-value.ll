; RUN: opt -passes=check-debugify < %s 2>&1 | FileCheck %s

define <2 x i64> @test-fun(<2 x i64> %A) !dbg !6 {
  %and = and <2 x i64> %A, <i64 23, i64 42>, !dbg !14

; CHECK: ERROR: dbg.value operand has size 128, but its variable has size 256
  call void @llvm.dbg.value(metadata <2 x i64> %and, metadata !12, metadata !DIExpression()), !dbg !15

; CHECK: ERROR: dbg.value operand has size 512, but its variable has size 256
  call void @llvm.dbg.value(metadata <2 x i256> zeroinitializer, metadata !12, metadata !DIExpression()), !dbg !15

; CHECK: ERROR: dbg.value operand has size 8, but its variable has size 256
  call void @llvm.dbg.value(metadata i8 0, metadata !12, metadata !DIExpression()), !dbg !15

; Assume the debugger implicitly uses the lower 256 bits.
; CHECK-NOT: ERROR: dbg.value operand has size 512, but its variable has size 256
  call void @llvm.dbg.value(metadata i512 0, metadata !12, metadata !DIExpression()), !dbg !15

; Assume the debugger implicitly zero-extends unsigned values.
; CHECK-NOT: ERROR: dbg.value operand has size 4, but its variable has size 32
  call void @llvm.dbg.value(metadata i8 0, metadata !17, metadata !DIExpression()), !dbg !15

  ret <2 x i64> %and, !dbg !16
}
; CHECK: CheckModuleDebugify: FAIL

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!3, !4}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "/Users/vsk/src/llvm.org-main/llvm/test/DebugInfo/debugify-bogus-dbg-value.ll", directory: "/")
!2 = !{}
!3 = !{i32 4}
!4 = !{i32 4}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "test-fun", linkageName: "test-fun", scope: null, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !0, retainedNodes: !8)
!7 = !DISubroutineType(types: !2)
!8 = !{!9, !11, !12}
!9 = !DILocalVariable(name: "1", scope: !6, file: !1, line: 1, type: !10)
!10 = !DIBasicType(name: "ty64", size: 64, encoding: DW_ATE_signed)
!11 = !DILocalVariable(name: "2", scope: !6, file: !1, line: 2, type: !10)
!12 = !DILocalVariable(name: "3", scope: !6, file: !1, line: 3, type: !13)

; Set the size here to an incorrect value to check that the size diagnostic
; triggers.
!13 = !DIBasicType(name: "ty128", size: 256, encoding: DW_ATE_signed)

!14 = !DILocation(line: 2, column: 1, scope: !6)
!15 = !DILocation(line: 3, column: 1, scope: !6)
!16 = !DILocation(line: 4, column: 1, scope: !6)

!17 = !DILocalVariable(name: "4", scope: !6, file: !1, line: 3, type: !18)
!18 = !DIBasicType(name: "ty32_unsigned", size: 32, encoding: DW_ATE_unsigned)
