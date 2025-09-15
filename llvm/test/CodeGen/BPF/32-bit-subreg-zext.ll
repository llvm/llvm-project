; RUN: llc -O2 -mtriple=bpfel -mattr=+alu32 < %s | FileCheck %s
; RUN: llc -O2 -mtriple=bpfel -mcpu=v3 < %s | FileCheck %s
; RUN: llc -O2 -mtriple=bpfeb -mattr=+alu32 < %s | FileCheck %s
; RUN: llc -O2 -mtriple=bpfeb -mcpu=v3 < %s | FileCheck %s
;
; long zext(unsigned int a)
; {
;   long b = a;
;   return b;
; }

; Function Attrs: norecurse nounwind
define dso_local i64 @zext(i32 %a) local_unnamed_addr #0 {
entry:
  %conv = zext i32 %a to i64
  ; CHECK-NOT: r[[#]] <<= 32
  ; CHECK-NOT: r[[#]] >>= 32
  ret i64 %conv
}

attributes #0 = { norecurse nounwind }
