; Ensure that LowerTypeTests control flow simplification correctly handle phi nodes.
; RUN: opt -S -lowertypetests -lowertypetests-summary-action=import -lowertypetests-read-summary=%S/Inputs/import.yaml < %s | FileCheck %s

target datalayout = "e-p:64:64"

declare i1 @llvm.type.test(i8* %ptr, metadata %bitset) nounwind readnone

; CHECK: define i1 @bytearray7(i8* [[p:%.*]])
define i1 @bytearray7(i8* %p) {
  %x = call i1 @llvm.type.test(i8* %p, metadata !"bytearray7")
  br i1 %x, label %t, label %f

t:
  br label %f

f:
  ; CHECK: %test = phi i1 [ false, %{{[0-9]+}} ], [ true, %t ], [ false, %0 ]
  %test = phi i1 [ false, %0 ], [ true, %t ]
  ret i1 %test
}
