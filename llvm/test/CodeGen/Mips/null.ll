; RUN: llc  -march=mipsel -mcpu=mips16 < %s | FileCheck %s -check-prefix=16


define i32 @main() nounwind {
entry:
  ret i32 0

; 16: 	.set	mips16                  # @main


; 16:	jrc	$ra

}
