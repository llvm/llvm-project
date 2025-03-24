; RUN: llc -mtriple=mips -mattr=-long-calls %s -o - \
; RUN:   | FileCheck -check-prefix=OFF %s
; RUN: llc -mtriple=mips -mattr=+long-calls,+noabicalls %s -o - \
; RUN:   | FileCheck -check-prefix=ON32 %s

; RUN: llc -mtriple=mips -mattr=+long-calls,-noabicalls %s -o - \
; RUN:   | FileCheck -check-prefix=OFF %s

; RUN: llc -mtriple=mips64 -target-abi n32 -mattr=-long-calls %s -o - \
; RUN:   | FileCheck -check-prefix=OFF %s
; RUN: llc -mtriple=mips64 -target-abi n32 -mattr=+long-calls,+noabicalls %s -o - \
; RUN:   | FileCheck -check-prefix=ON32 %s

; RUN: llc -mtriple=mips64 -target-abi n64 -mattr=-long-calls %s -o - \
; RUN:   | FileCheck -check-prefix=OFF %s
; RUN: llc -mtriple=mips64 -target-abi n64 -mattr=+long-calls,+noabicalls %s -o - \
; RUN:   | FileCheck -check-prefix=ON64 %s

declare void @callee()
declare void @llvm.memset.p0.i32(ptr nocapture writeonly, i8, i32, i1)

@val = internal unnamed_addr global [20 x i32] zeroinitializer, align 4

define void @caller() {

; Use `jal` instruction with R_MIPS_26 relocation.
; OFF: jal callee
; OFF: jal memset

; Save the `callee` and `memset` addresses in $25 register
; and use `jalr` for the jumps.
; ON32: lui    $1, %hi(callee)
; ON32: addiu  $25, $1, %lo(callee)
; ON32: jalr   $25

; ON32: lui    $1, %hi(memset)
; ON32: addiu  $25, $1, %lo(memset)
; ON32: jalr   $25

; ON64: lui     $1, %highest(callee)
; ON64: daddiu  $1, $1, %higher(callee)
; ON64: daddiu  $1, $1, %hi(callee)
; ON64: daddiu  $25, $1, %lo(callee)
; ON64: jalr    $25

; ON64: lui     $2, %highest(memset)
; ON64: daddiu  $1, $2, %higher(memset)
; ON64: dsll    $1, $1, 16
; ON64: daddiu  $1, $1, %hi(memset)
; ON64: dsll    $1, $1, 16
; ON64: daddiu  $25, $1, %lo(memset)
; ON64: jalr    $25

  call void @callee()
  call void @llvm.memset.p0.i32(ptr align 4 @val, i8 0, i32 80, i1 false)
  ret  void
}
