; RUN: llc < %s -mtriple=bpf -mcpu=v1 -verify-machineinstrs | FileCheck %s

; Source code:
; struct xdp_md {
;   unsigned data;
;   unsigned data_end;
; };
;
; int gbl;
; int xdp_dummy(struct xdp_md *xdp)
; {
;   char tmp;
;   long addr;
;
;   if (gbl) {
;     long addr1 = (long)xdp->data;
;     tmp = *(char *)addr1;
;     if (tmp == 1)
;       return 3;
;   } else {
;     tmp = *(volatile char *)(long)xdp->data_end;
;     if (tmp == 1)
;       return 2;
;   }
;   addr = (long)xdp->data;
;   tmp = *(volatile char *)addr;
;   if (tmp == 0)
;     return 1;
;   return 0;
; }

%struct.xdp_md = type { i32, i32 }

@gbl = common local_unnamed_addr global i32 0, align 4

; Function Attrs: norecurse nounwind
define i32 @xdp_dummy(ptr nocapture readonly %xdp) local_unnamed_addr #0 {
entry:
  %0 = load i32, ptr @gbl, align 4
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  %1 = load i32, ptr %xdp, align 4
  %conv = zext i32 %1 to i64
  %2 = inttoptr i64 %conv to ptr
  %3 = load i8, ptr %2, align 1
  %cmp = icmp eq i8 %3, 1
  br i1 %cmp, label %cleanup20, label %if.end12
; CHECK:  r1 = *(u32 *)(r1 + 0)
; CHECK:  r2 = *(u8 *)(r1 + 0)

if.else:                                          ; preds = %entry
  %data_end = getelementptr inbounds %struct.xdp_md, ptr %xdp, i64 0, i32 1
  %4 = load i32, ptr %data_end, align 4
  %conv6 = zext i32 %4 to i64
; CHECK:  r2 = *(u32 *)(r1 + 4)
  %5 = inttoptr i64 %conv6 to ptr
  %6 = load volatile i8, ptr %5, align 1
  %cmp8 = icmp eq i8 %6, 1
  br i1 %cmp8, label %cleanup20, label %if.else.if.end12_crit_edge

if.else.if.end12_crit_edge:                       ; preds = %if.else
  %.pre = load i32, ptr %xdp, align 4
  br label %if.end12
; CHECK: r1 = *(u32 *)(r1 + 0)

if.end12:                                         ; preds = %if.else.if.end12_crit_edge, %if.then
  %7 = phi i32 [ %.pre, %if.else.if.end12_crit_edge ], [ %1, %if.then ]
  %conv14 = zext i32 %7 to i64
; CHECK-NOT: r1 <<= 32
; CHECK-NOT: r1 >>= 32
  %8 = inttoptr i64 %conv14 to ptr
  %9 = load volatile i8, ptr %8, align 1
; CHECK:  r1 = *(u8 *)(r1 + 0)
  %cmp16 = icmp eq i8 %9, 0
  %.28 = zext i1 %cmp16 to i32
  br label %cleanup20

cleanup20:                                        ; preds = %if.then, %if.end12, %if.else
  %retval.1 = phi i32 [ 3, %if.then ], [ 2, %if.else ], [ %.28, %if.end12 ]
  ret i32 %retval.1
}

attributes #0 = { norecurse nounwind }
