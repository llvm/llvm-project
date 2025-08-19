; RUN: llc -march=bpf -mcpu=v4 < %s | FileCheck %s
;
; Source code:
;    int bar(int a) {
;       __label__ l1, l2;
;       void * volatile tgt;
;       int ret = 0;
;       if (a)
;         tgt = &&l1; // synthetic jump table generated here
;       else
;         tgt = &&l2; // another synthetic jump table
;       goto *tgt;
;   l1: ret += 1;
;   l2: ret += 2;
;       return ret;
;     }
;
; Compilation Flags:
;   clang --target=bpf -mcpu=v4 -O2 -emit-llvm -S test.c

define dso_local range(i32 2, 4) i32 @bar(i32 noundef %a) local_unnamed_addr{
entry:
  %tgt = alloca ptr, align 8
  %tobool.not = icmp eq i32 %a, 0
  %. = select i1 %tobool.not, ptr blockaddress(@bar, %l2), ptr blockaddress(@bar, %l1)
  store volatile ptr %., ptr %tgt, align 8
  %tgt.0.tgt.0.tgt.0.tgt.0. = load volatile ptr, ptr %tgt, align 8
  indirectbr ptr %tgt.0.tgt.0.tgt.0.tgt.0., [label %l1, label %l2]

l1:                                               ; preds = %entry
  br label %l2

l2:                                               ; preds = %l1, %entry
  %ret.0 = phi i32 [ 3, %l1 ], [ 2, %entry ]
  ret i32 %ret.0
}

; CHECK:            r2 = BPF.JT.0.0 ll
; CHECK-NEXT:       r2 = *(u64 *)(r2 + 0)
; CHECK-NEXT:       r3 = BPF.JT.0.1 ll
; CHECK-NEXT:       r3 = *(u64 *)(r3 + 0)
; CHECK-NEXT:       if w1 == 0 goto LBB0_2
; CHECK:            r3 = r2
; CHECK-NEXT:  LBB0_2:                                 # %entry
; CHECK-NEXT:       *(u64 *)(r10 - 8) = r3
; CHECK-NEXT:       r1 = *(u64 *)(r10 - 8)
; CHECK-NEXT:       gotox r1
; CHECK-NEXT:  .Ltmp0:                                 # Block address taken
; CHECK-NEXT:  LBB0_3:                                 # %l1
; CHECK-NEXT:       w0 = 3
; CHECK-NEXT:       goto LBB0_5
; CHECK-NEXT:  .Ltmp1:                                 # Block address taken
; CHECK-NEXT:  LBB0_4:                                 # %l2
; CHECK-NEXT:       w0 = 2
; CHECK-NEXT:  LBB0_5:                                 # %.split
; CHECK-NEXT:       exit
;
; CHECK:            .section        .jumptables,"",@progbits
; CHECK-NEXT:  BPF.JT.0.0:
; CHECK-NEXT:       .quad   LBB0_3
; CHECK-NEXT:       .size   BPF.JT.0.0, 8
; CHECK-NEXT:  BPF.JT.0.1:
; CHECK-NEXT:       .quad   LBB0_4
; CHECK-NEXT:       .size   BPF.JT.0.1, 8
