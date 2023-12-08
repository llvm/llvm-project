; RUN: opt < %s -passes='print<delinearization>' -disable-output
; REQUIRES: asserts

; Test that SCEV divide code doesn't crash when attempting to create a SCEV
; with operands of different types. In this case, the visitAddRecExpr
; function tries to create an AddRec where the start and step are different
; types.

target datalayout = "e-m:e-p:32:32-i64:64-a:0-v32:32-n16:32"

define fastcc void @test() {
entry:
  %0 = load i16, ptr undef, align 2
  %conv21 = zext i16 %0 to i32
  br label %for.cond7.preheader

for.cond7.preheader:
  %p1.022 = phi ptr [ undef, %entry ], [ %add.ptr, %for.end ]
  br label %for.body11

for.body11:
  %arrayidx.phi = phi ptr [ %p1.022, %for.cond7.preheader ], [ undef, %for.body11 ]
  store i8 undef, ptr %arrayidx.phi, align 1
  br i1 undef, label %for.body11, label %for.end

for.end:
  %add.ptr = getelementptr inbounds i8, ptr %p1.022, i32 %conv21
  br label %for.cond7.preheader
}
