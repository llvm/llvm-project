; RUN: llc -O2 -march=bpfel -mattr=+alu32 < %s | FileCheck %s
;
; unsigned int select_cc_32 (unsigned a, unsigned b, int c, int d)
; {
;   if (a > b)
;     return c;
;   else
;     return d;
; }
;
; long long select_cc_32_64 (unsigned a, unsigned b, long long c, long long d)
; {
;   if (a > b)
;     return c;
;   else
;     return d;
; }
;
; int select_cc_64_32 (long long a, long long b, int c, int d)
; {
;   if (a > b)
;     return c;
;   else
;     return d;
; }
;
; int selecti_cc_32 (unsigned a, int c, int d)
; {
;   if (a > 10)
;     return c;
;   else
;     return d;
; }
;
; long long selecti_cc_32_64 (unsigned a, long long c, long long d)
; {
;   if (a > 11)
;     return c;
;   else
;     return d;
; }
;
; int selecti_cc_64_32 (long long a, int c, int d)
; {
;   if (a > 12)
;     return c;
;   else
;     return d;
; }

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @select_cc_32(i32 %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
entry:
  %cmp = icmp ugt i32 %a, %b
  %c.d = select i1 %cmp, i32 %c, i32 %d
  ret i32 %c.d
}
; CHECK-LABEL: select_cc_32
; CHECK: r{{[0-9]+}} = w{{[0-9]+}}
; CHECK-NOT: r{{[0-9]+}} <<= 32
; CHECK-NOT: r{{[0-9]+}} >>= 32

; Function Attrs: norecurse nounwind readnone
define dso_local i64 @select_cc_32_64(i32 %a, i32 %b, i64 %c, i64 %d) local_unnamed_addr #0 {
entry:
  %cmp = icmp ugt i32 %a, %b
  %c.d = select i1 %cmp, i64 %c, i64 %d
  ret i64 %c.d
}
; CHECK-LABEL: select_cc_32_64
; CHECK: r{{[0-9]+}} = w{{[0-9]+}}
; CHECK-NOT: r{{[0-9]+}} <<= 32
; CHECK-NOT: r{{[0-9]+}} >>= 32

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @select_cc_64_32(i64 %a, i64 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
entry:
  %cmp = icmp sgt i64 %a, %b
  %c.d = select i1 %cmp, i32 %c, i32 %d
  ret i32 %c.d
}
; CHECK-LABEL: select_cc_64_32
; CHECK-NOT: r{{[0-9]+}} <<= 32

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @selecti_cc_32(i32 %a, i32 %c, i32 %d) local_unnamed_addr #0 {
entry:
  %cmp = icmp ugt i32 %a, 10
  %c.d = select i1 %cmp, i32 %c, i32 %d
  ret i32 %c.d
}
; CHECK-LABEL: selecti_cc_32
; CHECK: r{{[0-9]+}} = w{{[0-9]+}}
; CHECK-NOT: r{{[0-9]+}} <<= 32
; CHECK-NOT: r{{[0-9]+}} >>= 32

; Function Attrs: norecurse nounwind readnone
define dso_local i64 @selecti_cc_32_64(i32 %a, i64 %c, i64 %d) local_unnamed_addr #0 {
entry:
  %cmp = icmp ugt i32 %a, 11
  %c.d = select i1 %cmp, i64 %c, i64 %d
  ret i64 %c.d
}
; CHECK-LABEL: selecti_cc_32_64
; CHECK: r{{[0-9]+}} = w{{[0-9]+}}
; CHECK-NOT: r{{[0-9]+}} <<= 32
; CHECK-NOT: r{{[0-9]+}} >>= 32

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @selecti_cc_64_32(i64 %a, i32 %c, i32 %d) local_unnamed_addr #0 {
entry:
  %cmp = icmp sgt i64 %a, 12
  %c.d = select i1 %cmp, i32 %c, i32 %d
  ret i32 %c.d
}
; CHECK-LABEL: selecti_cc_64_32
; CHECK-NOT: r{{[0-9]+}} <<= 32
