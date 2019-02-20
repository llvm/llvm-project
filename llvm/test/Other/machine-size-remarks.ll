; REQUIRES: x86-registered-target
; RUN: llc -mtriple x86_64-apple-darwin %s -pass-remarks-analysis='size-info'\
; RUN: -pass-remarks-output=%t.yaml -o /dev/null < %s 2> %t; \
; RUN: cat %t %t.yaml | FileCheck %s

; Make sure that machine-level size remarks work.
; Test the following:
; - When we create a MachineFunction (e.g, during instruction selection), it
;   has a size of 0.
; - The initial size of the function after filling it is positive.
; - After that, we can increase or decrease the size of the function.
; - ... The final size must be positive.
; - ... The delta can be negative or positive.

; CHECK: remark: <unknown>:0:0: X86 DAG->DAG Instruction Selection: Function:
; CHECK-SAME: main: MI Instruction count changed from 0
; CHECK-SAME: to [[INIT:[1-9][0-9]*]]; Delta: [[INIT]]
; CHECK-NEXT: remark: <unknown>:0:0: Simple Register Coalescing: Function: main:
; CHECK-SAME: MI Instruction count changed from [[INIT]] to
; CHECK-SAME: [[FINAL:[1-9][0-9]*]];
; CHECK-SAME: Delta: [[DELTA:-?[1-9][0-9]*]]
; CHECK-NEXT: --- !Analysis
; CHECK-NEXT: Pass:            size-info
; CHECK-NEXT: Name:            FunctionMISizeChange
; CHECK-NEXT: Function:        main
; CHECK-NEXT: Args:
; CHECK-NEXT: - Pass:            'X86 DAG->DAG Instruction Selection'
; CHECK-NEXT: - String:          ': Function: '
; CHECK-NEXT: - Function:        main
; CHECK-NEXT: - String:          ': '
; CHECK-NEXT: - String:          'MI Instruction count changed from '
; CHECK-NEXT: - MIInstrsBefore:  '0'
; CHECK-NEXT:  - String:          ' to '
; CHECK-NEXT:  - MIInstrsAfter:   '[[INIT]]'
; CHECK-NEXT:  - String:          '; Delta: '
; CHECK-NEXT:  - Delta:           '[[INIT]]'
; CHECK-DAG: --- !Analysis
; CHECK-NEXT: Pass:            size-info
; CHECK-NEXT: Name:            FunctionMISizeChange
; CHECK-NEXT: Function:        main
; CHECK-NEXT: Args:
; CHECK-NEXT:   - Pass:            Simple Register Coalescing
; CHECK-NEXT:   - String:          ': Function: '
; CHECK-NEXT:   - Function:        main
; CHECK-NEXT:   - String:          ': '
; CHECK-NEXT:   - String:          'MI Instruction count changed from '
; CHECK-NEXT:   - MIInstrsBefore:  '[[INIT]]'
; CHECK-NEXT:   - String:          ' to '
; CHECK-NEXT:   - MIInstrsAfter:   '[[FINAL]]'
; CHECK-NEXT:   - String:          '; Delta: '
; CHECK-NEXT:   - Delta:           '[[DELTA]]'
define i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  ret i32 0
}

attributes #0 = { noinline nounwind optnone ssp uwtable }
