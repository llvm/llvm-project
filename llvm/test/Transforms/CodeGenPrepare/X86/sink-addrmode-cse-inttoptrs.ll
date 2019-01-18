; RUN: opt -mtriple=x86_64-- -codegenprepare                        %s -S -o - | FileCheck %s --check-prefix=CGP
; RUN: opt -mtriple=x86_64-- -codegenprepare -load-store-vectorizer %s -S -o - | FileCheck %s --check-prefix=LSV

; Make sure CodeGenPrepare doesn't emit multiple inttoptr instructions
; of the same integer value while sinking address computations, but
; rather CSEs them on the fly: excessive inttoptr's confuse SCEV
; into thinking that related pointers have nothing to do with each other.
;
; Triggering this problem involves having just right addressing modes,
; and verifying that the motivating pass (LoadStoreVectorizer) is able
; to benefit from it - just right LSV-policies. Hence the atypical combination
; of the target and datalayout / address spaces in this test.

target datalayout = "p1:32:32:32"

define void @main(i32 %tmp, i32 %off) {
; CGP:     = inttoptr
; CGP-NOT: = inttoptr
; LSV:     = load <2 x float>
; LSV:     = load <2 x float>
entry:
  %tmp1 = inttoptr i32 %tmp to float addrspace(1)*
  %arrayidx.i.7 = getelementptr inbounds float, float addrspace(1)* %tmp1, i32 %off
  %add20.i.7 = add i32 %off, 1
  %arrayidx22.i.7 = getelementptr inbounds float, float addrspace(1)* %tmp1, i32 %add20.i.7
  br label %for.body

for.body:
  %tmp8 = phi float [ undef, %entry ], [ %tmp62, %for.body ]
  %tmp28 = load float, float addrspace(1)* %arrayidx.i.7
  %tmp29 = load float, float addrspace(1)* %arrayidx22.i.7
  %arrayidx.i321.7 = getelementptr inbounds float, float addrspace(1)* %tmp1, i32 0
  %tmp43 = load float, float addrspace(1)* %arrayidx.i321.7
  %arrayidx22.i327.7 = getelementptr inbounds float, float addrspace(1)* %tmp1, i32 1
  %tmp44 = load float, float addrspace(1)* %arrayidx22.i327.7
  %tmp62 = tail call fast float @foo(float %tmp8, float %tmp44, float %tmp43, float %tmp29, float %tmp28)
  br label %for.body
}

declare float @foo(float, float, float, float, float)
