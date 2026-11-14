; RUN: opt %s -verify-debuginfo-preserve -debugify-report-dropped -passes='function(loop-mssa(licm))' -disable-output 2>&1 | FileCheck %s
; RUN: rm -f %t.json
; RUN: opt %s -verify-debuginfo-preserve -debugify-report-dropped -verify-di-preserve-export=%t.json -passes='function(loop-mssa(licm))' -disable-output
; RUN: FileCheck %s --check-prefix=JSON < %t.json

; CHECK: WARNING:{{.*(dropped|did not generate)}} DILocation for %{{.*}} = mul i32 %a, %b
; CHECK: CheckModuleDebugify (original debuginfo): FAIL

; JSON: "action":"{{drop|not-generate}}"
; JSON-SAME: "instr":"mul"

define i32 @hoist_mul(i32 %a, i32 %b, i1 %cond) !dbg !5 {
entry:
  br label %loop, !dbg !8

loop:
  %v = mul i32 %a, %b, !dbg !9
  br i1 %cond, label %loop, label %exit, !dbg !10

exit:
  ret i32 %v, !dbg !11
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "hoist_mul", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!6 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 2, column: 3, scope: !5)
!9 = !DILocation(line: 3, column: 5, scope: !5)
!10 = !DILocation(line: 4, column: 3, scope: !5)
!11 = !DILocation(line: 5, column: 3, scope: !5)
