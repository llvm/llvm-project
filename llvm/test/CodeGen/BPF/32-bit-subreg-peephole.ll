; RUN: llc -O2 -march=bpfel -mattr=+alu32 < %s | FileCheck %s
;
; long long select_u(unsigned a, unsigned b, long long c, long long d)
; {
;   if (a > b)
;     return c;
;   else
;     return d;
; }
;
; long long select_u_2(unsigned a, unsigned long long b, long long c, long long d)
; {
;   if (a > b)
;     return c;
;   else
;     return d;
; }
;
; long long select_s(signed a, signed b, long long c, long long d)
; {
;   if (a > b)
;     return c;
;   else
;     return d;
; }
;
; long long bar ();
;
; int foo (int b, int c)
; {
;   unsigned int i32_val = (unsigned int) bar();
;
;   if (i32_val < 10)
;     return b;
;   else
;     return c;
; }
;
; int *inc_p (int *p, unsigned a)
; {
;   return p + a;
; }

; Function Attrs: norecurse nounwind readnone
define dso_local i64 @select_u(i32 %a, i32 %b, i64 %c, i64 %d) local_unnamed_addr #0 {
; CHECK-LABEL: select_u:
entry:
  %cmp = icmp ugt i32 %a, %b
  %c.d = select i1 %cmp, i64 %c, i64 %d
; CHECK-NOT: r{{[0-9]+}} <<= 32
; CHECK-NOT: r{{[0-9]+}} >>= 32
; CHECK: if r{{[0-9]+}} {{<|>}} r{{[0-9]+}} goto
  ret i64 %c.d
}

; Function Attrs: norecurse nounwind readnone
define dso_local i64 @select_u_2(i32 %a, i64 %b, i64 %c, i64 %d) local_unnamed_addr #0 {
; CHECK-LABEL: select_u_2:
entry:
  %conv = zext i32 %a to i64
; CHECK-NOT: r{{[0-9]+}} <<= 32
; CHECK-NOT: r{{[0-9]+}} >>= 32
  %cmp = icmp ugt i64 %conv, %b
  %c.d = select i1 %cmp, i64 %c, i64 %d
  ret i64 %c.d
}

; Function Attrs: norecurse nounwind readnone
define dso_local i64 @select_s(i32 %a, i32 %b, i64 %c, i64 %d) local_unnamed_addr #0 {
; CHECK-LABEL: select_s:
entry:
  %cmp = icmp sgt i32 %a, %b
  %c.d = select i1 %cmp, i64 %c, i64 %d
; CHECK: r{{[0-9]+}} <<= 32
; CHECK-NEXT: r{{[0-9]+}} s>>= 32
; CHECK: if r{{[0-9]+}} s{{<|>}} r{{[0-9]+}} goto
  ret i64 %c.d
}

; Function Attrs: nounwind
define dso_local i32 @foo(i32 %b, i32 %c) local_unnamed_addr #0 {
; CHECK-LABEL: foo:
entry:
  %call = tail call i64 bitcast (i64 (...)* @bar to i64 ()*)() #2
  %conv = trunc i64 %call to i32
  %cmp = icmp ult i32 %conv, 10
; The shifts can't be optimized out because %call comes from function call
; returning i64 so the high bits might be valid.
; CHECK: r{{[0-9]+}} <<= 32
; CHECK-NEXT: r{{[0-9]+}} >>= 32
  %b.c = select i1 %cmp, i32 %b, i32 %c
; CHECK: if r{{[0-9]+}} {{<|>}} {{[0-9]+}} goto
  ret i32 %b.c
}

declare dso_local i64 @bar(...) local_unnamed_addr #1

; Function Attrs: norecurse nounwind readnone
define dso_local i32* @inc_p(i32* readnone %p, i32 %a) local_unnamed_addr #0 {
; CHECK-LABEL: inc_p:
entry:
  %idx.ext = zext i32 %a to i64
; CHECK-NOT: r{{[0-9]+}} <<= 32
; CHECK-NOT: r{{[0-9]+}} >>= 32
  %add.ptr = getelementptr inbounds i32, i32* %p, i64 %idx.ext
  ret i32* %add.ptr
}
