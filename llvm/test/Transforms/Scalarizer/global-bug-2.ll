; RUN: opt < %s -scalarizer -S -o - | FileCheck %s
; RUN: opt < %s -passes='function(scalarizer)' -S | FileCheck %s

; The scalarizer used to change the name of the global variable
; Check that the we don't do that any longer.
;
; CHECK: @c.a = global i16 0, align 1

@c.a = global i16 0, align 1

define void @c() {
entry:
  br label %for.cond1

for.cond1:                                        ; preds = %for.cond1, %entry
  %d.sroa.0.0 = phi <4 x i16*> [ <i16* @c.a, i16* @c.a, i16* @c.a, i16* @c.a>, %entry ], [ %d.sroa.0.1.vec.insert, %for.cond1 ]
  %d.sroa.0.0.vec.extract = extractelement <4 x i16*> %d.sroa.0.0, i32 0
  %d.sroa.0.1.vec.insert = shufflevector <4 x i16*> <i16* @c.a, i16* null, i16* undef, i16* undef>, <4 x i16*> %d.sroa.0.0, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  br label %for.cond1
}
