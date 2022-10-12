; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:     -mcpu=pwr9 -ppc-asm-full-reg-names < %s | FileCheck %s

@z = external local_unnamed_addr global ptr, align 8

; Function Attrs: norecurse nounwind readonly
define signext i32 @_Z2tcii(i32 signext %x, i32 signext %y) local_unnamed_addr #0 {
entry:
  %0 = load ptr, ptr @z, align 8
  %add = add nsw i32 %y, %x
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i32, ptr %0, i64 %idxprom
  %1 = load i32, ptr %arrayidx, align 4
  ret i32 %1
; CHECK-LABEL: @_Z2tcii
; CHECK: extswsli {{r[0-9]+}}, {{r[0-9]+}}, 2
}
