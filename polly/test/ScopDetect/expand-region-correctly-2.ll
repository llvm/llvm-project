; RUN: opt %loadNPMPolly '-passes=print<polly-detect>' -disable-output < %s 2>&1 | FileCheck %s
;
; CHECK: Valid Region for Scop: if.end.1631 => for.cond.1647.outer
;
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define void @qtm_decompress() #0 {
entry:
  br label %if.end.1631

if.end.1631:                                      ; preds = %entry
  br i1 false, label %for.cond.1647.preheader, label %if.then.1635

if.then.1635:                                     ; preds = %if.end.1631
  br label %for.cond.1647.preheader

for.cond.1647.preheader:                          ; preds = %if.then.1635, %if.end.1631
  br label %for.cond.1647.outer

for.cond.1647.outer:                              ; preds = %do.end.1685, %for.cond.1647.preheader
  %bits_needed.5.ph = phi i8 [ 8, %for.cond.1647.preheader ], [ undef, %do.end.1685 ]
  br label %for.cond.1647

for.cond.1647:                                    ; preds = %do.cond.1718, %for.cond.1647.outer
  %bits_needed.5 = phi i8 [ 8, %do.cond.1718 ], [ %bits_needed.5.ph, %for.cond.1647.outer ]
  br i1 undef, label %do.cond.1718, label %if.then.1659

if.then.1659:                                     ; preds = %for.cond.1647
  br i1 false, label %do.end.1685, label %cleanup.1785

do.end.1685:                                      ; preds = %if.then.1659
  br label %for.cond.1647.outer

do.cond.1718:                                     ; preds = %for.cond.1647
  br i1 false, label %land.lhs.true.1736, label %for.cond.1647

land.lhs.true.1736:                               ; preds = %do.cond.1718
  br label %if.then.1742

if.then.1742:                                     ; preds = %land.lhs.true.1736
  unreachable

cleanup.1785:                                     ; preds = %if.then.1659
  ret void
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.8.0 (trunk 250010) (llvm/trunk 250018)"}
