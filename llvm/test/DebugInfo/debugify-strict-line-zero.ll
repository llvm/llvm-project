; RUN: opt -passes=check-debugify -disable-output < %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
; RUN: opt -passes=check-debugify -debugify-strict-line-zero -disable-output < %s 2>&1 | FileCheck %s --check-prefix=STRICT

; DEFAULT-NOT: WARNING: Instruction with line 0 DebugLoc
; DEFAULT: CheckModuleDebugify: PASS

; STRICT-NOT: WARNING: Instruction with line 0 DebugLoc {{.*}} phi
; STRICT: WARNING: Instruction with line 0 DebugLoc in function foo --  ret i32 %v, !dbg !{{[0-9]+}}
; STRICT-NEXT: WARNING: Missing line 1
; STRICT-NEXT: CheckModuleDebugify: PASS

define i32 @foo(i1 %cond) !dbg !6 {
entry:
  br i1 %cond, label %then, label %merge, !dbg !8

then:
  br label %merge, !dbg !8

merge:
  %v = phi i32 [ 0, %entry ], [ 1, %then ], !dbg !9
  ret i32 %v, !dbg !9
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!3, !4}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{}
!3 = !{i32 2}
!4 = !{i32 0}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 2, column: 1, scope: !6)
!9 = !DILocation(line: 0, column: 1, scope: !6)
