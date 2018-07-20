; Ensure that IR count remarks in the legacy pass manager work.
; What this test should check for:
; * Positive, nonzero sizes before/after
; * Nonzero deltas
; * Sizes are being tracked properly across multiple remarks. E.g, if we have
;     original_count_1, final_count_1, and
;     original_count_2, final_count_2,
;  Then original_count_2 == final_count_1.

; For these remarks, the "function" field in the YAML file doesn't matter.
; Each of the testcases work by combining the output remarks with the
; optimization record emit using -pass-remarks-output. This is done to prevent
; test flakiness wrt instruction counts, but also ensure that the output values
; are equivalent in both outputs.

; RUN: opt < %s -inline -pass-remarks-analysis='size-info' \
; RUN: -pass-remarks-output=%t.yaml -S -o /dev/null 2> %t; \
; RUN: cat %t %t.yaml | FileCheck %s -check-prefix=CGSCC
; CGSCC: remark: <unknown>:0:0: Function Integration/Inlining:
; CGSCC-SAME: IR instruction count changed from
; CGSCC-SAME: [[ORIG:[1-9][0-9]*]] to [[FINAL:[1-9][0-9]*]];
; CGSCC-SAME: Delta: [[DELTA:-?[1-9][0-9]*]]
; CGSCC: --- !Analysis
; CGSCC-NEXT: Pass:            size-info
; CGSCC-NEXT: Name:            IRSizeChange
; CGSCC-NEXT: Function:
; CGSCC-NEXT: Args:            
; CGSCC-NEXT:  - Pass:            Function Integration/Inlining
; CGSCC-NEXT:  - String:          ': IR instruction count changed from '
; CGSCC-NEXT:  - IRInstrsBefore:  '[[ORIG]]'
; CGSCC-NEXT:  - String:          ' to '
; CGSCC-NEXT:  - IRInstrsAfter:   '[[FINAL]]'
; CGSCC-NEXT:  - String:          '; Delta: '
; CGSCC-NEXT:  - DeltaInstrCount: '[[DELTA]]'

; RUN: opt < %s -instcombine -pass-remarks-analysis='size-info' \
; RUN:-pass-remarks-output=%t.yaml -S -o /dev/null 2> %t; \
; RUN: cat %t %t.yaml | FileCheck %s -check-prefix=FUNC
; FUNC: remark: <unknown>:0:0: Combine redundant instructions:
; FUNC-SAME: IR instruction count changed from
; FUNC-SAME: [[SIZE1:[1-9][0-9]*]] to [[SIZE2:[1-9][0-9]*]];
; FUNC-SAME: Delta: [[DELTA1:-?[1-9][0-9]*]]
; FUNC-NEXT: remark: <unknown>:0:0: Combine redundant instructions:
; FUNC-SAME: IR instruction count changed from
; FUNC-SAME: [[SIZE2]] to [[SIZE3:[1-9][0-9]*]];
; FUNC-SAME: Delta: [[DELTA2:-?[1-9][0-9]*]]
; FUNC: --- !Analysis
; FUNC-NEXT: Pass:            size-info
; FUNC-NEXT: Name:            IRSizeChange
; FUNC-NEXT: Function:
; FUNC-NEXT: Args:            
; FUNC-NEXT:  - Pass:            Combine redundant instructions
; FUNC-NEXT:  - String:          ': IR instruction count changed from '
; FUNC-NEXT:  - IRInstrsBefore:  '[[SIZE1]]'
; FUNC-NEXT:  - String:          ' to '
; FUNC-NEXT:  - IRInstrsAfter:   '[[SIZE2]]'
; FUNC-NEXT:  - String:          '; Delta: '
; FUNC-NEXT:  - DeltaInstrCount: '[[DELTA1]]'
; FUNC: --- !Analysis
; FUNC-NEXT: Pass:            size-info
; FUNC-NEXT: Name:            IRSizeChange
; FUNC-NEXT: Function:
; FUNC-NEXT: Args:            
; FUNC-NEXT:   - Pass:            Combine redundant instructions
; FUNC-NEXT:   - String:          ': IR instruction count changed from '
; FUNC-NEXT:   - IRInstrsBefore:  '[[SIZE2]]'
; FUNC-NEXT:   - String:          ' to '
; FUNC-NEXT:   - IRInstrsAfter:   '[[SIZE3]]'
; FUNC-NEXT:   - String:          '; Delta: '
; FUNC-NEXT:   - DeltaInstrCount: '[[DELTA2]]'

; RUN: opt < %s -globaldce -pass-remarks-analysis='size-info' \
; RUN: -pass-remarks-output=%t.yaml -S -o /dev/null 2> %t; \
; RUN: cat %t %t.yaml | FileCheck %s -check-prefix=MODULE
; MODULE: remark:
; MODULE-SAME: Dead Global Elimination:
; MODULE-SAME: IR instruction count changed from
; MODULE-SAME: [[ORIG:[1-9][0-9]*]] to [[FINAL:[1-9][0-9]*]];
; MODULE-SAME: Delta: [[DELTA:-?[1-9][0-9]*]]
; MODULE: --- !Analysis
; MODULE-NEXT: Pass:            size-info
; MODULE-NEXT: Name:            IRSizeChange
; MODULE-NEXT: Function:
; MODULE-NEXT: Args:            
; MODULE-NEXT:   - Pass:            Dead Global Elimination
; MODULE-NEXT:   - String:          ': IR instruction count changed from '
; MODULE-NEXT:   - IRInstrsBefore:  '[[ORIG]]'
; MODULE-NEXT:   - String:          ' to '
; MODULE-NEXT:   - IRInstrsAfter:   '[[FINAL]]'
; MODULE-NEXT:   - String:          '; Delta: '
; MODULE-NEXT:   - DeltaInstrCount: '[[DELTA]]'

; RUN: opt < %s -dce -pass-remarks-analysis='size-info' \
; RUN: -pass-remarks-output=%t.yaml -S -o /dev/null 2> %t; \
; RUN: cat %t %t.yaml | FileCheck %s -check-prefix=BB
; BB: remark: <unknown>:0:0: Dead Code Elimination:
; BB-SAME: IR instruction count changed from
; BB-SAME: [[ORIG:[1-9][0-9]*]] to [[FINAL:[1-9][0-9]*]];
; BB-SAME: Delta: [[DELTA:-?[1-9][0-9]*]]
; BB: --- !Analysis
; BB-NEXT: Pass:            size-info
; BB-NEXT: Name:            IRSizeChange
; BB-NEXT: Function:
; BB-NEXT: Args:            
; BB-NEXT:   - Pass:            Dead Code Elimination
; BB-NEXT:   - String:          ': IR instruction count changed from '
; BB-NEXT:   - IRInstrsBefore:  '[[ORIG]]'
; BB-NEXT:   - String:          ' to '
; BB-NEXT:   - IRInstrsAfter:   '[[FINAL]]'
; BB-NEXT:   - String:          '; Delta: '
; BB-NEXT:   - DeltaInstrCount: '[[DELTA]]'

; RUN: opt < %s -loop-unroll -pass-remarks-analysis='size-info' \
; RUN: -pass-remarks-output=%t.yaml -S -o /dev/null 2> %t; \
; RUN: cat %t %t.yaml | FileCheck %s -check-prefix=LOOP
; LOOP: remark: <unknown>:0:0: Unroll loops:
; LOOP-SAME: IR instruction count changed from
; LOOP-SAME: [[ORIG:[1-9][0-9]*]] to [[FINAL:[1-9][0-9]*]];
; LOOP-SAME: Delta: [[DELTA:-?[1-9][0-9]*]]
; LOOP: --- !Analysis
; LOOP-NEXT: Pass:            size-info
; LOOP-NEXT: Name:            IRSizeChange
; LOOP-NEXT: Function:
; LOOP-NEXT: Args:            
; LOOP-DAG:   - Pass:            Unroll loops
; LOOP-NEXT:   - String:          ': IR instruction count changed from '
; LOOP-NEXT:   - IRInstrsBefore:  '[[ORIG]]'
; LOOP-NEXT:   - String:          ' to '
; LOOP-NEXT:   - IRInstrsAfter:   '[[FINAL]]'
; LOOP-NEXT:   - String:          '; Delta: '
; LOOP-NEXT:   - DeltaInstrCount: '[[DELTA]]'
declare i1 ()* @boop()

define internal i1 @pluto() {
  %F = call i1 ()* () @boop( )
  %c = icmp eq i1 ()* %F, @pluto
  ret i1 %c
}

define i32 @foo(i32 %x) {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  %0 = load i32, i32* %x.addr, align 4
  ret i32 %0
}

define i32 @bar(i32 %x) {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  %0 = load i32, i32* %x.addr, align 4
  %call = call i32 @foo(i32 %0)
  br label %for.body
for.body:
  %s.06 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %i.05 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %add = add nsw i32 %i.05, 4
  %inc = add nsw i32 %i.05, 1
  %exitcond = icmp eq i32 %inc, 16
  br i1 %exitcond, label %for.end, label %for.body
for.end:
  ret i32 %add
}
