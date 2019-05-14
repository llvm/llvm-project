; RUN: opt %s -keep-loops=false -switch-to-lookup=true -simplifycfg -S | FileCheck %s
; RUN: opt %s -passes='simplify-cfg<no-keep-loops;switch-to-lookup>' -S | FileCheck %s

define void @f6() #0 {
; CHECK-LABEL: entry:

entry:
  br label %for.cond.i

for.cond.i:                                       ; preds = %f1.exit.i, %entry
  switch i16 undef, label %f1.exit.i [
    i16 -1, label %cond.false.i3.i
    i16 1, label %cond.false.i3.i
    i16 0, label %cond.false.i3.i
  ]

cond.false.i3.i:                                  ; preds = %for.cond.i, %for.cond.i, %for.cond.i
  br label %f1.exit.i

f1.exit.i:                                        ; preds = %cond.false.i3.i, %for.cond.i
  %cond.i4.i = phi i16 [ undef, %cond.false.i3.i ], [ 1, %for.cond.i ]
  %tobool7.i = icmp ne i16 %cond.i4.i, 0
  br label %for.cond.i
}
