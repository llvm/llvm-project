; RUN: llc -mattr=sram,eijmpcall < %s -march=avr | FileCheck %s

@brind.k = private unnamed_addr constant [2 x i8*] [i8* blockaddress(@brind, %return), i8* blockaddress(@brind, %b)], align 1

define i8 @brind(i8 %p) {
; CHECK-LABEL: brind:
; CHECK: ld r30
; CHECK: ldd r31
; CHECK: ijmp
entry:
  %idxprom = sext i8 %p to i16
  %arrayidx = getelementptr inbounds [2 x i8*], [2 x i8*]* @brind.k, i16 0, i16 %idxprom
  %s = load i8*, i8** %arrayidx
  indirectbr i8* %s, [label %return, label %b]
b:
  br label %return
return:
  %retval.0 = phi i8 [ 4, %b ], [ 2, %entry ]
  ret i8 %retval.0
}
