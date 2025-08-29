; RUN: llc -march=bpf -mcpu=v4 < %s | FileCheck %s
;
; Source code:
;   int foo(unsigned a) {
;     __label__ l1, l2;
;     void *jt1[] = {[0]=&&l1, [1]=&&l2};
;     int ret = 0;
;
;     goto *jt1[a % 2];
;     l1: ret += 1;
;     l2: ret += 3;
;     return ret;
;   }
;
; Compilation Flags:
;   clang --target=bpf -mcpu=v4 -O2 -emit-llvm -S test.c

@__const.foo.jt1 = private unnamed_addr constant [2 x ptr] [ptr blockaddress(@foo, %l1), ptr blockaddress(@foo, %l2)], align 8

define dso_local range(i32 3, 5) i32 @foo(i32 noundef %a) local_unnamed_addr {
entry:
  %rem = and i32 %a, 1
  %idxprom = zext nneg i32 %rem to i64
  %arrayidx = getelementptr inbounds nuw [2 x ptr], ptr @__const.foo.jt1, i64 0, i64 %idxprom
  %0 = load ptr, ptr %arrayidx, align 8
  indirectbr ptr %0, [label %l1, label %l2]

l1:                                               ; preds = %entry
  br label %l2

l2:                                               ; preds = %l1, %entry
  %ret.0 = phi i32 [ 4, %l1 ], [ 3, %entry ]
  ret i32 %ret.0
}

; CHECK:            w1 &= 1
; CHECK-NEXT:       r1 <<= 3
; CHECK-NEXT:       r2 = BPF.JT.0.0 ll
; CHECK-NEXT:       r2 += r1
; CHECK-NEXT:       r1 = *(u64 *)(r2 + 0)
; CHECK-NEXT:       gotox r1
; CHECK-NEXT:  .Ltmp0:                              # Block address taken
; CHECK-NEXT:  LBB0_1:                              # %l1
; CHECK-NEXT:       w0 = 4
; CHECK-NEXT:       goto LBB0_3
; CHECK-NEXT:  .Ltmp1:                              # Block address taken
; CHECK-NEXT:  LBB0_2:                              # %l2
; CHECK-NEXT:       w0 = 3
; CHECK-NEXT:  LBB0_3:                              # %.split
; CHECK-NEXT:       exit
;
; CHECK:            .section        .jumptables,"",@progbits
; CHECK-NEXT:  BPF.JT.0.0:
; CHECK-NEXT:       .quad   LBB0_1
; CHECK-NEXT:       .quad   LBB0_2
; CHECK-NEXT:       .size   BPF.JT.0.0, 16
