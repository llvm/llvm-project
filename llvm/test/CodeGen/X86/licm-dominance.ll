; RUN: llc -asm-verbose=true < %s | FileCheck %s

; MachineLICM should check dominance before hoisting instructions.
; CHECK: ## in Loop:
; CHECK-NEXT:	xorb	%al, %al
; CHECK-NEXT:	testb	%al, %al

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.2"

define void @CMSColorWorldCreateParametricData() nounwind uwtable optsize ssp {
entry:
  br label %for.body.i

for.body.i:                                       
  br i1 undef, label %for.inc.i, label %if.then26.i

if.then26.i:                                      
  br i1 undef, label %if.else.i.i, label %lor.lhs.false.i.i

if.else.i.i:                                      
  br i1 undef, label %lor.lhs.false.i.i, label %if.then116.i.i

lor.lhs.false.i.i:                                
  br i1 undef, label %for.inc.i, label %if.then116.i.i

if.then116.i.i:                                   
  unreachable

for.inc.i:                                        
  %cmp17.i = icmp ult i64 undef, undef
  br i1 %cmp17.i, label %for.body.i, label %if.end28.i

if.end28.i:                                       
  ret void
}
