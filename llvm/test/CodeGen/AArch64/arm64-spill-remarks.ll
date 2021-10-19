; RUN: llc < %s -mtriple=arm64-apple-ios7.0 -aarch64-neon-syntax=apple -pass-remarks-missed=regalloc 2>&1 | FileCheck -check-prefix=REMARK %s
; RUN: llc < %s -mtriple=arm64-apple-ios7.0 -aarch64-neon-syntax=apple -pass-remarks-missed=regalloc -pass-remarks-with-hotness 2>&1 | FileCheck -check-prefix=HOTNESS %s
; RUN: llc < %s -mtriple=arm64-apple-ios7.0 -aarch64-neon-syntax=apple 2>&1 | FileCheck -check-prefix=NO_REMARK %s
; RUN: llc < %s -mtriple=arm64-apple-ios7.0 -aarch64-neon-syntax=apple -pass-remarks-output=%t.yaml -pass-remarks-with-hotness 2>&1 | FileCheck -check-prefix=NO_REMARK %s
; RUN: cat %t.yaml | FileCheck -check-prefix=YAML %s
;
; Verify that remarks below the hotness threshold are not output.
; RUN: llc < %s -mtriple=arm64-apple-ios7.0 -aarch64-neon-syntax=apple -pass-remarks-missed=regalloc \
; RUN:       -pass-remarks-with-hotness -pass-remarks-hotness-threshold=500 \
; RUN:       2>&1 | FileCheck -check-prefix=THRESHOLD %s
; RUN: llc < %s -mtriple=arm64-apple-ios7.0 -aarch64-neon-syntax=apple -pass-remarks-output=%t.threshold.yaml \
; RUN:       -pass-remarks-with-hotness -pass-remarks-hotness-threshold=500 \
; RUN:       2>&1 | FileCheck -check-prefix=NO_REMARK %s
; RUN: cat %t.threshold.yaml | FileCheck -check-prefix=THRESHOLD_YAML %s

; This has two nested loops, each with one value that has to be spilled and
; then reloaded.

; (loop3:)
; REMARK: remark: /tmp/kk.c:3:20: 1 spills 1.000000e+02 total spills cost 1 reloads 1.000000e+02 total reloads cost generated in loop{{$}}
; (loop2:)
; REMARK: remark: /tmp/kk.c:2:20: 1 spills 1.000000e+04 total spills cost 1 reloads 1.000000e+04 total reloads cost generated in loop{{$}}
; (loop:)
; REMARK: remark: /tmp/kk.c:1:20: 2 spills 1.010000e+04 total spills cost 2 reloads 1.010000e+04 total reloads cost generated in loop{{$}}
; (func:)
; REMARK: remark: /tmp/kk.c:1:1: 3 spills 1.020000e+04 total spills cost 3 reloads 1.020000e+04 total reloads cost generated in function

; (loop3:)
; HOTNESS: remark: /tmp/kk.c:3:20: 1 spills 1.000000e+02 total spills cost 1 reloads 1.000000e+02 total reloads cost generated in loop (hotness: 300)
; (loop2:)
; HOTNESS: remark: /tmp/kk.c:2:20: 1 spills 1.000000e+04 total spills cost 1 reloads 1.000000e+04 total reloads cost generated in loop (hotness: 30000)
; (loop:)
; HOTNESS: remark: /tmp/kk.c:1:20: 2 spills 1.010000e+04 total spills cost 2 reloads 1.010000e+04 total reloads cost generated in loop (hotness: 300)

; NO_REMARK-NOT: remark

; THRESHOLD-NOT: (hotness: 300)
; THRESHOLD: remark: /tmp/kk.c:2:20: 1 spills 1.000000e+04 total spills cost 1 reloads 1.000000e+04 total reloads cost generated in loop (hotness: 30000)

; YAML: --- !Missed
; YAML: Pass:            regalloc
; YAML: Name:            LoopSpillReloadCopies
; YAML: DebugLoc:        { File: '/tmp/kk.c', Line: 3, Column: 20 }
; YAML: Function:        fpr128
; YAML: Hotness:         300
; YAML: Args:
; YAML:   - NumSpills:       '1'
; YAML:   - String:          ' spills '
; YAML:   - TotalSpillsCost: '1.000000e+02'
; YAML:   - String:          ' total spills cost '
; YAML:   - NumReloads:      '1'
; YAML:   - String:          ' reloads '
; YAML:   - TotalReloadsCost: '1.000000e+02'
; YAML:   - String:          ' total reloads cost '
; YAML:   - String:          generated in loop
; YAML: ...
; YAML: --- !Missed
; YAML: Pass:            regalloc
; YAML: Name:            LoopSpillReloadCopies
; YAML: DebugLoc:        { File: '/tmp/kk.c', Line: 2, Column: 20 }
; YAML: Function:        fpr128
; YAML: Hotness:         30000
; YAML: Args:
; YAML:   - NumSpills:       '1'
; YAML:   - String:          ' spills '
; YAML:   - TotalSpillsCost: '1.000000e+04'
; YAML:   - String:          ' total spills cost '
; YAML:   - NumReloads:      '1'
; YAML:   - String:          ' reloads '
; YAML:   - TotalReloadsCost: '1.000000e+04'
; YAML:   - String:          ' total reloads cost '
; YAML:   - String:          generated in loop
; YAML: ...
; YAML: --- !Missed
; YAML: Pass:            regalloc
; YAML: Name:            LoopSpillReloadCopies
; YAML: DebugLoc:        { File: '/tmp/kk.c', Line: 1, Column: 20 }
; YAML: Function:        fpr128
; YAML: Hotness:         300
; YAML: Args:
; YAML:   - NumSpills:       '2'
; YAML:   - String:          ' spills '
; YAML:   - TotalSpillsCost: '1.010000e+04'
; YAML:   - String:          ' total spills cost '
; YAML:   - NumReloads:      '2'
; YAML:   - String:          ' reloads '
; YAML:   - TotalReloadsCost: '1.010000e+04'
; YAML:   - String:          ' total reloads cost '
; YAML:   - String:          generated in loop
; YAML: ...
; YAML: --- !Missed
; YAML: Pass:            regalloc
; YAML: Name:            SpillReloadCopies
; YAML: DebugLoc:        { File: '/tmp/kk.c', Line: 1, Column: 1 }
; YAML: Function:        fpr128
; YAML: Hotness:         3
; YAML: Args:
; YAML:   - NumSpills:       '3'
; YAML:   - String:          ' spills '
; YAML:   - TotalSpillsCost: '1.020000e+04'
; YAML:   - String:          ' total spills cost '
; YAML:   - NumReloads:      '3'
; YAML:   - String:          ' reloads '
; YAML:   - TotalReloadsCost: '1.020000e+04'
; YAML:   - String:          ' total reloads cost '
; YAML:   - String:          generated in function
; YAML: ...

; THRESHOLD_YAML-NOT: Hotness:         300{{$}}
; THRESHOLD_YAML: --- !Missed
; THRESHOLD_YAML: Pass:            regalloc
; THRESHOLD_YAML: Name:            LoopSpillReloadCopies
; THRESHOLD_YAML: DebugLoc:        { File: '/tmp/kk.c', Line: 2, Column: 20 }
; THRESHOLD_YAML: Function:        fpr128
; THRESHOLD_YAML: Hotness:         30000
; THRESHOLD_YAML: Args:
; THRESHOLD_YAML:   - NumSpills:       '1'
; THRESHOLD_YAML:   - String:          ' spills '
; THRESHOLD_YAML:   - TotalSpillsCost: '1.000000e+04'
; THRESHOLD_YAML:   - String:          ' total spills cost '
; THRESHOLD_YAML:   - NumReloads:      '1'
; THRESHOLD_YAML:   - String:          ' reloads '
; THRESHOLD_YAML:   - TotalReloadsCost: '1.000000e+04'
; THRESHOLD_YAML:   - String:          ' total reloads cost '
; THRESHOLD_YAML:   - String:          generated in loop
; THRESHOLD_YAML: ...

define void @fpr128(<4 x float>* %p) nounwind ssp !prof !11 !dbg !6 {
entry:
  br label %loop, !dbg !8

loop:
  %i = phi i32 [ 0, %entry], [ %i.2, %end2 ]
  br label %loop2, !dbg !9

loop2:
  %j = phi i32 [ 0, %loop], [ %j.2, %loop2 ]
  call void asm sideeffect "; inlineasm", "~{q0},~{q1},~{q2},~{q3},~{q4},~{q5},~{q6},~{q7},~{q8},~{q9},~{q10},~{q11},~{q12},~{q13},~{q14},~{q15},~{q16},~{q17},~{q18},~{q19},~{q20},~{q21},~{q22},~{q23},~{q24},~{q25},~{q26},~{q27},~{q28},~{q29},~{q30},~{q31},~{x0},~{x1},~{x2},~{x3},~{x4},~{x5},~{x6},~{x7},~{x8},~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x15},~{x16},~{x17},~{x18},~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28},~{fp},~{lr},~{sp},~{memory}"() nounwind
  %j.2 = add i32 %j, 1
  %c2 = icmp slt i32 %j.2, 100
  br i1 %c2, label %loop2, label %end2, !prof !12

end2:
  call void asm sideeffect "; inlineasm", "~{q0},~{q1},~{q2},~{q3},~{q4},~{q5},~{q6},~{q7},~{q8},~{q9},~{q10},~{q11},~{q12},~{q13},~{q14},~{q15},~{q16},~{q17},~{q18},~{q19},~{q20},~{q21},~{q22},~{q23},~{q24},~{q25},~{q26},~{q27},~{q28},~{q29},~{q30},~{q31},~{x0},~{x1},~{x2},~{x3},~{x4},~{x5},~{x6},~{x7},~{x8},~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x15},~{x16},~{x17},~{x18},~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28},~{fp},~{lr},~{sp},~{memory}"() nounwind
  %i.2 = add i32 %i, 1
  %c = icmp slt i32 %i.2, 100
  br i1 %c, label %loop, label %end, !prof !12

end:
  br label %loop3

loop3:
  %k = phi i32 [ 0, %end], [ %k.2, %loop3 ]
  call void asm sideeffect "; inlineasm", "~{q0},~{q1},~{q2},~{q3},~{q4},~{q5},~{q6},~{q7},~{q8},~{q9},~{q10},~{q11},~{q12},~{q13},~{q14},~{q15},~{q16},~{q17},~{q18},~{q19},~{q20},~{q21},~{q22},~{q23},~{q24},~{q25},~{q26},~{q27},~{q28},~{q29},~{q30},~{q31},~{x0},~{x1},~{x2},~{x3},~{x4},~{x5},~{x6},~{x7},~{x8},~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x15},~{x16},~{x17},~{x18},~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28},~{fp},~{lr},~{sp},~{memory}"() nounwind
  %k.2 = add i32 %k, 1
  %c3 = icmp slt i32 %k.2, 100
  br i1 %c3, label %loop3, label %end3, !dbg !10, !prof !12

end3:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!1 = !DIFile(filename: "/tmp/kk.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"PIC Level", i32 2}
!5 = !{!"clang version 3.9.0 "}
!6 = distinct !DISubprogram(name: "success", scope: !1, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 1, column: 20, scope: !6)
!9 = !DILocation(line: 2, column: 20, scope: !6)
!10 = !DILocation(line: 3, column: 20, scope: !6)
!11 = !{!"function_entry_count", i64 3}
!12 = !{!"branch_weights", i32 99, i32 1}
