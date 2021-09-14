; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=nanomips -verify-machineinstrs --stop-after=finalize-isel < %s | FileCheck %s --check-prefix=AFTER-ISEL

@brind_opts = constant [2 x i8*] [i8* blockaddress(@brind, %block1), i8* blockaddress(@brind, %block2)]

define i8 @brind(i8 %p) {
  %index = sext i8 %p to i16
  %element = getelementptr inbounds [2 x i8*], [2 x i8*]* @brind_opts, i16 0, i16 %index
  %address = load i8*, i8** %element
; AFTER-ISEL: PseudoIndirectBranchNM
; CHECK: jrc $t4
; CHECK: JRC_NM
  indirectbr i8* %address, [label %block1, label %block2]
block1:
  ret i8 23
block2:
  ret i8 37
}
