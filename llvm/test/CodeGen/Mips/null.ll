; RUN: llc  -march=mipsel -mattr=mips16 < %s | FileCheck %s -check-prefix=16


define i32 @main() nounwind {
entry:
  ret i32 0

; 16: 	.set	mips16


; 16:	jrc	$ra

}
