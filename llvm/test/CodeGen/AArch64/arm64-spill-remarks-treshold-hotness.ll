; RUN: llc < %s -mtriple=arm64-apple-ios7.0 -aarch64-neon-syntax=apple -pass-remarks-missed=regalloc \
; RUN:       -pass-remarks-with-hotness 2>&1 | FileCheck %s

; RUN: llc < %s -mtriple=arm64-apple-ios7.0 -aarch64-neon-syntax=apple -pass-remarks-missed=regalloc \
; RUN:       -pass-remarks-with-hotness -pass-remarks-hotness-threshold=1 \
; RUN:       2>&1 | FileCheck -check-prefix=THRESHOLD %s

; CHECK: remark: /tmp/kk.c:3:20: 1 spills 3.200000e+01 total spills cost 1 reloads 3.200000e+01 total reloads cost generated in loop{{$}}
; THRESHOLD-NOT: remark

define void @fpr128(ptr %p) nounwind ssp {
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
  br i1 %c2, label %loop2, label %end2

end2:
  call void asm sideeffect "; inlineasm", "~{q0},~{q1},~{q2},~{q3},~{q4},~{q5},~{q6},~{q7},~{q8},~{q9},~{q10},~{q11},~{q12},~{q13},~{q14},~{q15},~{q16},~{q17},~{q18},~{q19},~{q20},~{q21},~{q22},~{q23},~{q24},~{q25},~{q26},~{q27},~{q28},~{q29},~{q30},~{q31},~{x0},~{x1},~{x2},~{x3},~{x4},~{x5},~{x6},~{x7},~{x8},~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x15},~{x16},~{x17},~{x18},~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28},~{fp},~{lr},~{sp},~{memory}"() nounwind
  %i.2 = add i32 %i, 1
  %c = icmp slt i32 %i.2, 100
  br i1 %c, label %loop, label %end

end:
  br label %loop3

loop3:
  %k = phi i32 [ 0, %end], [ %k.2, %loop3 ]
  call void asm sideeffect "; inlineasm", "~{q0},~{q1},~{q2},~{q3},~{q4},~{q5},~{q6},~{q7},~{q8},~{q9},~{q10},~{q11},~{q12},~{q13},~{q14},~{q15},~{q16},~{q17},~{q18},~{q19},~{q20},~{q21},~{q22},~{q23},~{q24},~{q25},~{q26},~{q27},~{q28},~{q29},~{q30},~{q31},~{x0},~{x1},~{x2},~{x3},~{x4},~{x5},~{x6},~{x7},~{x8},~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x15},~{x16},~{x17},~{x18},~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28},~{fp},~{lr},~{sp},~{memory}"() nounwind
  %k.2 = add i32 %k, 1
  %c3 = icmp slt i32 %k.2, 100
  br i1 %c3, label %loop3, label %end3, !dbg !10

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
