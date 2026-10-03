; RUN: opt -passes=mergefunc -S < %s | FileCheck %s --implicit-check-not='@second'

; Assignment IDs link a store to the debug records of the function that
; contains it. Keep the IDs of the surviving function.

; CHECK-LABEL: define void @first(
; CHECK-NEXT: store i8 %v, ptr %p, align 1, !DIAssignID ![[ID:[0-9]+]]
; CHECK-NEXT: #dbg_assign(i8 %v, ![[VAR:[0-9]+]], !DIExpression(), ![[ID]], ptr %p, !DIExpression(), ![[LOC:[0-9]+]])
; CHECK-NEXT: ret void
; CHECK: ![[SP:[0-9]+]] = distinct !DISubprogram(name: "first",
; CHECK: ![[ID]] = distinct !DIAssignID()
; CHECK: ![[VAR]] = !DILocalVariable(name: "value", scope: ![[SP]],
; CHECK: ![[LOC]] = !DILocation(line: 1, scope: ![[SP]])
define void @first(ptr %p, i8 %v) !dbg !5 {
  store i8 %v, ptr %p, !DIAssignID !9
  #dbg_assign(i8 %v, !7, !DIExpression(), !9, ptr %p, !DIExpression(), !11)
  ret void
}

define internal void @second(ptr %p, i8 %v) !dbg !6 {
  store i8 %v, ptr %p, !DIAssignID !10
  #dbg_assign(i8 %v, !8, !DIExpression(), !10, ptr %p, !DIExpression(), !12)
  ret void
}

define void @calls(ptr %p, i8 %v) {
  call void @first(ptr %p, i8 %v)
  call void @second(ptr %p, i8 %v)
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "merge.c", directory: "/")
!2 = !DISubroutineType(types: !13)
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!5 = distinct !DISubprogram(name: "first", scope: !1, file: !1, line: 1, type: !2, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = distinct !DISubprogram(name: "second", scope: !1, file: !1, line: 2, type: !2, scopeLine: 2, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!7 = !DILocalVariable(name: "value", scope: !5, file: !1, line: 1, type: !14)
!8 = !DILocalVariable(name: "value", scope: !6, file: !1, line: 2, type: !14)
!9 = distinct !DIAssignID()
!10 = distinct !DIAssignID()
!11 = !DILocation(line: 1, scope: !5)
!12 = !DILocation(line: 2, scope: !6)
!13 = !{}
!14 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
