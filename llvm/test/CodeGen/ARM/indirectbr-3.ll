; REQUIRES: asserts
; RUN: llc < %s -mtriple=thumbv7-apple-ios -arm-atomic-cfg-tidy=0 -stats 2>&1 | FileCheck %s

; If ARMBaseInstrInfo::AnalyzeBlocks returns the wrong value, which was possible
; for blocks with indirect branches, the IfConverter could end up deleting
; blocks that were the destinations of indirect branches, leaving branches to
; nowhere.
; <rdar://problem/14464830>

define i32 @preserve_blocks(i32 %x) {
; preserve_blocks:
; CHECK: Block address taken
; CHECK: movs r0, #1
; CHECK: Block address taken
; CHECK: movs r0, #2
; CHECK-NOT: Address of block that was removed by CodeGen

; Separate bug. There are no valid diamonds to if-convert in this file.
; There was a bug in the if-conversion code that would if-convert a false
; diamond where one side had a return and the other had an indirect branch.
; Make sure no diamond conversions occurred while compiling this file.
; CHECK: Statistics Collected
; CHECK-NOT: 1 ifcvt          - Number of diamond if-conversions performed
entry:
  %c2 = icmp slt i32 %x, 3
  %blockaddr = select i1 %c2, i8* blockaddress(@preserve_blocks, %ibt1), i8* blockaddress(@preserve_blocks, %ibt2)
  %c1 = icmp eq i32 %x, 0
  br i1 %c1, label %pre_ib, label %nextblock

nextblock:
  ret i32 3

ibt1:
  ret i32 2

ibt2:
  ret i32 1

pre_ib:
  indirectbr i8* %blockaddr, [ label %ibt1, label %ibt2 ]
}
