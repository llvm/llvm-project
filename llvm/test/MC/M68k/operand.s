# RUN: llvm-mc -triple=m68k -show-inst-operands %s 2> %t0
# RUN: FileCheck %s < %t0

; CHECK:	parsed instruction: [token 'move.l', immediate 123, %24]
move.l	#123, %d0
