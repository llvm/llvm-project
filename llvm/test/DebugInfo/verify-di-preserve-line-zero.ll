; RUN: opt %s -verify-debuginfo-preserve -passes='function(loop-mssa(licm))' -disable-output 2>&1 | FileCheck %s --check-prefix=DEFAULT
; RUN: opt %s -verify-debuginfo-preserve -debugify-strict-line-zero -passes='function(loop-mssa(licm))' -disable-output 2>&1 | FileCheck %s --check-prefix=STRICT
; RUN: opt %s -verify-each-debuginfo-preserve -debugify-strict-line-zero -passes='function(dce,instcombine)' -disable-output 2>&1 | FileCheck %s --check-prefix=EACH
; RUN: rm -f %t.json
; RUN: opt %s -verify-debuginfo-preserve -debugify-strict-line-zero -verify-di-preserve-export=%t.json -passes='function(loop-mssa(licm))' -disable-output
; RUN: FileCheck %s --check-prefix=JSON < %t.json

; DEFAULT-NOT: WARNING:
; DEFAULT: CheckModuleDebugify (original debuginfo): PASS

; STRICT-NOT: preexisting_zero
; STRICT: WARNING:  degraded DILocation to line 0 for   %{{.*}} = call i32 @pure_fn(i32 %a), !dbg !{{[0-9]+}} (BB: exit, Fn: hoist_call, File: test.c)
; STRICT: CheckModuleDebugify (original debuginfo): FAIL

; EACH-NOT: WARNING:
; EACH: DCEPass: PASS
; EACH: InstCombinePass: PASS

; JSON: "action":"line-zero"
; JSON-SAME: "fn-name":"hoist_call"
; JSON-SAME: "instr":"call"

define i32 @pure_fn(i32 %x) nounwind willreturn memory(none) !dbg !14 {
  ret i32 %x, !dbg !15
}

define i32 @preexisting_zero(i32 %x) !dbg !5 {
  %r = add i32 %x, 1, !dbg !8
  ret i32 %r, !dbg !9
}

define i32 @hoist_call(i32 %a, i1 %cond) !dbg !7 {
entry:
  br label %loop, !dbg !10

loop:
  %v = call i32 @pure_fn(i32 %a), !dbg !11
  br i1 %cond, label %loop, label %exit, !dbg !12

exit:
  ret i32 %v, !dbg !13
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "preexisting_zero", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!6 = !DISubroutineType(types: !2)
!7 = distinct !DISubprogram(name: "hoist_call", scope: !1, file: !1, line: 5, type: !6, scopeLine: 5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!8 = !DILocation(line: 0, column: 1, scope: !5)
!9 = !DILocation(line: 2, column: 3, scope: !5)
!10 = !DILocation(line: 6, column: 3, scope: !7)
!11 = !DILocation(line: 7, column: 5, scope: !7)
!12 = !DILocation(line: 8, column: 3, scope: !7)
!13 = !DILocation(line: 9, column: 3, scope: !7)
!14 = distinct !DISubprogram(name: "pure_fn", scope: !1, file: !1, line: 10, type: !6, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!15 = !DILocation(line: 11, column: 3, scope: !14)
