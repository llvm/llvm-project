; RUN: llc -march=mips -target-abi o32 --mattr=+long-calls,+noabicalls < %s \
; RUN:   | FileCheck -check-prefix=O32 %s
; RUN: llc -march=mips -target-abi o32 --mattr=-long-calls,+noabicalls < %s \
; RUN:   | FileCheck -check-prefix=O32 %s
; RUN: llc -march=mips64 -target-abi n64 --mattr=+long-calls,+noabicalls < %s \
; RUN:   | FileCheck -check-prefix=N64 %s
; RUN: llc -march=mips64 -target-abi n64 --mattr=-long-calls,+noabicalls < %s \
; RUN:   | FileCheck -check-prefix=N64 %s

declare void @far() #0

define void @near() #1 {
  ret void
}

define void @foo() {
  call void @far()

; O32-LABEL: foo:
; O32:         lui     $1, %hi(far)
; O32-NEXT:    addiu   $25, $1, %lo(far)
; O32-NEXT:    jalr    $25

; N64-LABEL: foo:
; N64:         lui     $1, %highest(far)
; N64-NEXT:    daddiu  $1, $1, %higher(far)
; N64-NEXT:    dsll    $1, $1, 16
; N64-NEXT:    daddiu  $1, $1, %hi(far)
; N64-NEXT:    dsll    $1, $1, 16
; N64-NEXT:    daddiu  $25, $1, %lo(far)
; N64-NEXT:    jalr    $25

  call void @near()

; O32:         jal near
; N64:         jal near

  ret void
}

attributes #0 = { "long-call" }
attributes #1 = { "short-call" }
