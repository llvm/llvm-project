; RUN: opt < %s -loop-unswitch -enable-new-pm=0 -disable-output
; RUN: opt < %s -loop-unswitch -enable-new-pm=0 -enable-mssa-loop-dependency=true -verify-memoryssa -disable-output

define void @test1(i32* %S2) {
entry:
	br i1 false, label %list_Length.exit, label %cond_true.i
cond_true.i:		; preds = %entry
	ret void
list_Length.exit:		; preds = %entry
	br i1 false, label %list_Length.exit9, label %cond_true.i5
cond_true.i5:		; preds = %list_Length.exit
	ret void
list_Length.exit9:		; preds = %list_Length.exit
	br i1 false, label %bb78, label %return
bb44:		; preds = %bb78, %cond_next68
	br i1 %tmp49.not, label %bb62, label %bb62.loopexit
bb62.loopexit:		; preds = %bb44
	br label %bb62
bb62:		; preds = %bb62.loopexit, %bb44
	br i1 false, label %return.loopexit, label %cond_next68
cond_next68:		; preds = %bb62
	br i1 false, label %return.loopexit, label %bb44
bb78:		; preds = %list_Length.exit9
	%tmp49.not = icmp eq i32* %S2, null		; <i1> [#uses=1]
	br label %bb44
return.loopexit:		; preds = %cond_next68, %bb62
	%retval.0.ph = phi i32 [ 1, %cond_next68 ], [ 0, %bb62 ]		; <i32> [#uses=1]
	br label %return
return:		; preds = %return.loopexit, %list_Length.exit9
	%retval.0 = phi i32 [ 0, %list_Length.exit9 ], [ %retval.0.ph, %return.loopexit ]		; <i32> [#uses=0]
	ret void
}

define void @test2() nounwind {
entry:
  br label %bb.nph

bb.nph:                                           ; preds = %entry
  %and.i13521 = and <4 x i1> undef, undef         ; <<4 x i1>> [#uses=1]
  br label %for.body

for.body:                                         ; preds = %for.body, %bb.nph
  %or.i = select <4 x i1> %and.i13521, <4 x i32> undef, <4 x i32> undef ; <<4 x i32>> [#uses=0]
  br i1 false, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; PR6879
define i32* @test3(i32** %p_45, i16 zeroext %p_46, i64 %p_47, i64 %p_48, i16 signext %p_49) nounwind {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.cond4, %entry
  br i1 false, label %for.cond4, label %for.end88

for.cond4:                                        ; preds = %for.cond
  %conv46 = trunc i32 0 to i8                     ; <i8> [#uses=2]
  %cmp60 = icmp sgt i8 %conv46, 124               ; <i1> [#uses=1]
  %or.cond = and i1 undef, %cmp60                 ; <i1> [#uses=1]
  %cond = select i1 %or.cond, i8 %conv46, i8 undef ; <i8> [#uses=0]
  br label %for.cond

for.end88:                                        ; preds = %for.cond
  ret i32* undef
}
