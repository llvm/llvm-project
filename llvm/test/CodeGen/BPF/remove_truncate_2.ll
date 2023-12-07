; RUN: llc < %s -march=bpf -verify-machineinstrs | FileCheck %s

; Source code:
; struct xdp_md {
;   unsigned data;
;   unsigned data_end;
; };
;
; int gbl;
; int xdp_dummy(struct xdp_md *xdp)
; {
;   char addr = *(char *)(long)xdp->data;
;   if (gbl) {
;     if (gbl == 1)
;       return 1;
;     if (addr == 1)
;       return 3;
;   } else if (addr == 0)
;     return 2;
;   return 0;
; }

%struct.xdp_md = type { i32, i32 }

@gbl = common local_unnamed_addr global i32 0, align 4

; Function Attrs: norecurse nounwind readonly
define i32 @xdp_dummy(ptr nocapture readonly %xdp) local_unnamed_addr #0 {
entry:
  %0 = load i32, ptr %xdp, align 4
  %conv = zext i32 %0 to i64
  %1 = inttoptr i64 %conv to ptr
  %2 = load i8, ptr %1, align 1
; CHECK:  r1 = *(u32 *)(r1 + 0)
; CHECK:  r1 = *(u8 *)(r1 + 0)
  %3 = load i32, ptr @gbl, align 4
  switch i32 %3, label %if.end [
    i32 0, label %if.else
    i32 1, label %cleanup
  ]

if.end:                                           ; preds = %entry
  %cmp4 = icmp eq i8 %2, 1
; CHECK:  r0 = 3
; CHECK-NOT:  r1 &= 255
; CHECK:  if r1 == 1 goto
  br i1 %cmp4, label %cleanup, label %if.end13

if.else:                                          ; preds = %entry
  %cmp9 = icmp eq i8 %2, 0
; CHECK:  r0 = 2
; CHECK-NOT:  r1 &= 255
; CHECK:  if r1 == 0 goto
  br i1 %cmp9, label %cleanup, label %if.end13

if.end13:                                         ; preds = %if.else, %if.end
  br label %cleanup

cleanup:                                          ; preds = %if.else, %if.end, %entry, %if.end13
  %retval.0 = phi i32 [ 0, %if.end13 ], [ 1, %entry ], [ 3, %if.end ], [ 2, %if.else ]
  ret i32 %retval.0
}

attributes #0 = { norecurse nounwind readonly }
