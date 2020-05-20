; RUN: llc -mtriple=aarch64---  --verify-machineinstrs -stop-before=finalize-isel -simplify-mir -o - < %s | FileCheck %s

; Here we check thatt the noredzone attribute is carried through the machine
; IR generation and is put in MachineFunctionInfo

define void @baz() {
  entry:
    ; CHECK-LABEL: name:            baz
    ; CHECK: machineFunctionInfo: {}
    ret void
}

define void @bar() #0 {
  entry:
    ; CHECK-LABEL: name:            bar
    ; CHECK: machineFunctionInfo:
    ; CHECK-NEXT: hasRedZone:      false
    ret void
}

attributes #0 = { noredzone }
