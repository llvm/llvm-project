; RUN: llc -march=x86-64 < %s
define i32 @signbitl(x86_fp80 %x) nounwind uwtable readnone {
entry:
  %tmp4 = bitcast x86_fp80 %x to i80
  %tmp4.lobit = lshr i80 %tmp4, 79
  %tmp = trunc i80 %tmp4.lobit to i32
  ret i32 %tmp
}
