; RUN: opt < %s -basic-aa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin13.4.0"

; CHECK-LABEL: compute1
; CHECK: MayAlias:	i32* %arrayidx8, i32* %out
; CHECK: MayAlias:	i32* %arrayidx11, i32* %out
; CHECK: MayAlias:	i32* %arrayidx11, i32* %arrayidx8
; CHECK: MayAlias:	i32* %arrayidx14, i32* %out
; CHECK: MayAlias:	i32* %arrayidx14, i32* %arrayidx8
; CHECK: MayAlias:	i32* %arrayidx11, i32* %arrayidx14
define void @compute1(i32 %num.0.lcssa, i32* %out) {
  %idxprom = zext i32 %num.0.lcssa to i64
  %arrayidx8 = getelementptr inbounds i32, i32* %out, i64 %idxprom
  %add9 = or i32 %num.0.lcssa, 1
  %idxprom10 = zext i32 %add9 to i64
  %arrayidx11 = getelementptr inbounds i32, i32* %out, i64 %idxprom10
  %add12 = or i32 %num.0.lcssa, 2
  %idxprom13 = zext i32 %add12 to i64
  %arrayidx14 = getelementptr inbounds i32, i32* %out, i64 %idxprom13
  ret void
}

; CHECK-LABEL: compute2
; CHECK: MayAlias: i32* %arrayidx11, i32* %out.addr
define void @compute2(i32 %num, i32* %out.addr) {
  %add9 = add i32 %num, 1
  %idxprom10 = zext i32 %add9 to i64
  %arrayidx11 = getelementptr inbounds i32, i32* %out.addr, i64 %idxprom10
  ret void
}
