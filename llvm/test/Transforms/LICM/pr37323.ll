;RUN: opt -verify-dom-info -passes='loop-simplify,require<postdomtree>,require<opt-remark-emit>,loop-mssa(licm),function(adce)' -S -o - %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

@c = external global i16, align 1

;Make sure this test do not crash while accessing PostDomTree which is not
;preserved in LICM.
;
;CHECK-LABEL: fn1(i1 %arg)
;CHECK-LABEL: for.cond.loopexit.split.loop.exit
;CHECK-LABEL: for.cond.loopexit.split.loop.exit1
define void @fn1(i1 %arg) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %if.end, %for.cond1, %entry
  %0 = phi i16 [ undef, %entry ], [ ptrtoint (ptr @c to i16), %if.end ], [ %.mux, %for.cond1 ]
  br i1 %arg, label %for.cond1, label %for.end8

for.cond1:                                        ; preds = %if.end, %for.cond
  %.mux = select i1 undef, i16 undef, i16 ptrtoint (ptr @c to i16)
  br i1 %arg, label %for.cond, label %if.end

if.end:                                           ; preds = %for.cond1
  br i1 %arg, label %for.cond, label %for.cond1

for.end8:                                         ; preds = %for.cond
  ret void
}
