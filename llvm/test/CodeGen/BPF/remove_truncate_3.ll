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
;
; unsigned int rol32(unsigned int word, unsigned int shift)
; {
;   return (word << shift) | (word >> ((-shift) & 31));
; }
%struct.xdp_md = type { i32, i32 }

@gbl = common local_unnamed_addr global i32 0, align 4

; Function Attrs: norecurse nounwind
define i32 @xdp_dummy(%struct.xdp_md* nocapture readonly) local_unnamed_addr #0 {
  %2 = load i32, i32* @gbl, align 4
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %11, label %4

; <label>:4:                                      ; preds = %1
  %5 = getelementptr inbounds %struct.xdp_md, %struct.xdp_md* %0, i64 0, i32 0
  %6 = load i32, i32* %5, align 4
  %7 = zext i32 %6 to i64
  %8 = inttoptr i64 %7 to i8*
  %9 = load i8, i8* %8, align 1
  %10 = icmp eq i8 %9, 1
  br i1 %10, label %28, label %23
; CHECK:  r1 = *(u32 *)(r1 + 0)
; CHECK:  r2 = *(u8 *)(r1 + 0)

; <label>:11:                                     ; preds = %1
  %12 = getelementptr inbounds %struct.xdp_md, %struct.xdp_md* %0, i64 0, i32 1
  %13 = load i32, i32* %12, align 4
  %14 = zext i32 %13 to i64
; CHECK:  r2 = *(u32 *)(r1 + 4)
  %15 = inttoptr i64 %14 to i8*
  %16 = load volatile i8, i8* %15, align 1
; CHECK:  r2 = *(u8 *)(r2 + 0)
  %17 = icmp eq i8 %16, 1
  br i1 %17, label %28, label %18

; <label>:18:                                     ; preds = %11
  %19 = getelementptr inbounds %struct.xdp_md, %struct.xdp_md* %0, i64 0, i32 0
  %20 = load i32, i32* %19, align 4
  %21 = zext i32 %20 to i64
  %22 = inttoptr i64 %21 to i8*
  br label %23
; CHECK: r1 = *(u32 *)(r1 + 0)

; <label>:23:                                     ; preds = %18, %4
  %24 = phi i8* [ %22, %18 ], [ %8, %4 ]
; CHECK-NOT: r1 <<= 32
; CHECK-NOT: r1 >>= 32
  %25 = load volatile i8, i8* %24, align 1
; CHECK:  r1 = *(u8 *)(r1 + 0)
  %26 = icmp eq i8 %25, 0
  %27 = zext i1 %26 to i32
  br label %28

; <label>:28:                                     ; preds = %4, %23, %11
  %29 = phi i32 [ 3, %4 ], [ 2, %11 ], [ %27, %23 ]
  ret i32 %29
}

; Function Attrs: norecurse nounwind readnone
define i32 @rol32(i32, i32) local_unnamed_addr #1 {
  %3 = shl i32 %0, %1
; CHECK: r3 <<= 32
; CHECK: r3 >>= 32
  %4 = sub i32 0, %1
  %5 = and i32 %4, 31
  %6 = lshr i32 %0, %5
; CHECK: r1 <<= 32
; CHECK: r1 >>= 32
  %7 = or i32 %6, %3
  ret i32 %7
}

attributes #0 = { norecurse nounwind }
attributes #1 = { norecurse nounwind readnone }
