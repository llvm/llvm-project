; Check call to mcount in case of long/short call options.
; RUN: llc -march=mips -target-abi o32 --mattr=+long-calls,+noabicalls < %s \
; RUN:   -mips-jalr-reloc=false | FileCheck -check-prefixes=CHECK,LONG %s
; RUN: llc -march=mips -target-abi o32 --mattr=-long-calls,+noabicalls < %s \
; RUN:   -mips-jalr-reloc=false | FileCheck -check-prefixes=CHECK,SHORT %s

define void @foo() {
entry:
  call void @_mcount()
  ret void

; CHECK-LABEL: foo
; LONG:          lui     $1, %hi(_mcount)
; LONG-NEXT:     addiu   $25, $1, %lo(_mcount)
; LONG-NEXT:     jalr    $25
; SHORT:         jal     _mcount
}

declare void @_mcount()
