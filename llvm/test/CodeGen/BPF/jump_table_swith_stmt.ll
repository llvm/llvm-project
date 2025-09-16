; RUN: llc -march=bpf -mcpu=v4 -bpf-min-jump-table-entries=3 < %s | FileCheck %s
;
; Source code:
;   int ret_user;
;   int foo(int a)
;   {
;      switch (a) {
;      case 1: ret_user = 18; break;
;      case 20: ret_user = 6; break;
;      case 30: ret_user = 2; break;
;      default: break;
;      }
;      return 0;
;   }
;
; Compilation Flags:
;   clang --target=bpf -mcpu=v4 -O2 -emit-llvm -S test.c

@ret_user = dso_local local_unnamed_addr global i32 0, align 4

define dso_local noundef i32 @foo(i32 noundef %a) local_unnamed_addr {
entry:
  switch i32 %a, label %sw.epilog [
    i32 1, label %sw.epilog.sink.split
    i32 20, label %sw.bb1
    i32 30, label %sw.bb2
  ]

sw.bb1:                                           ; preds = %entry
  br label %sw.epilog.sink.split

sw.bb2:                                           ; preds = %entry
  br label %sw.epilog.sink.split

sw.epilog.sink.split:                             ; preds = %entry, %sw.bb1, %sw.bb2
  %.sink = phi i32 [ 2, %sw.bb2 ], [ 6, %sw.bb1 ], [ 18, %entry ]
  store i32 %.sink, ptr @ret_user, align 4
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.epilog.sink.split, %entry
  ret i32 0
}

; CHECK:            w1 += -1
; CHECK-NEXT:       if w1 > 29 goto LBB0_5
; CHECK:            w2 = 18
; CHECK-NEXT:       r1 <<= 3
; CHECK-NEXT:       r3 = BPF.JT.0.0 ll
; CHECK-NEXT:       r4 = BPF.JT.0.0 ll
; CHECK-NEXT:       r4 += r1
; CHECK-NEXT:       r1 = *(u64 *)(r4 + 0)
; CHECK-NEXT:       r3 += r1
; CHECK-NEXT:       gotox r3
; CHECK-NEXT:  LBB0_2:                                 # %sw.bb1
; CHECK-NEXT:       w2 = 6
; CHECK-NEXT:       goto LBB0_4
; CHECK-NEXT:  LBB0_3:                                 # %sw.bb2
; CHECK-NEXT:       w2 = 2
; CHECK-NEXT:  LBB0_4:                                 # %sw.epilog.sink.split
; CHECK-NEXT:       r1 = ret_user ll
; CHECK-NEXT:       *(u32 *)(r1 + 0) = w2
; CHECK-NEXT:  LBB0_5:                                 # %sw.epilog
; CHECK-NEXT:       w0 = 0
; CHECK-NEXT:       exit

; CHECK:            .section        .jumptables,"",@progbits
; CHECK-NEXT:  BPF.JT.0.0:
; CHECK-NEXT:       .quad   LBB0_4
; CHECK-NEXT:       .quad   LBB0_5
; CHECK-NEXT:       .quad   LBB0_5
; CHECK-NEXT:       .quad   LBB0_5
; CHECK-NEXT:       .quad   LBB0_5
; CHECK-NEXT:       .quad   LBB0_5
; CHECK-NEXT:       .quad   LBB0_5
; CHECK-NEXT:       .quad   LBB0_5
; CHECK-NEXT:       .quad   LBB0_5
; CHECK-NEXT:       .quad   LBB0_5
; CHECK-NEXT:       .quad   LBB0_5
; CHECK-NEXT:       .quad   LBB0_5
; CHECK-NEXT:       .quad   LBB0_5
; CHECK-NEXT:       .quad   LBB0_5
; CHECK-NEXT:       .quad   LBB0_5
; CHECK-NEXT:       .quad   LBB0_5
; CHECK-NEXT:       .quad   LBB0_5
; CHECK-NEXT:       .quad   LBB0_5
; CHECK-NEXT:       .quad   LBB0_5
; CHECK-NEXT:       .quad   LBB0_2
; CHECK-NEXT:       .quad   LBB0_5
; CHECK-NEXT:       .quad   LBB0_5
; CHECK-NEXT:       .quad   LBB0_5
; CHECK-NEXT:       .quad   LBB0_5
; CHECK-NEXT:       .quad   LBB0_5
; CHECK-NEXT:       .quad   LBB0_5
; CHECK-NEXT:       .quad   LBB0_5
; CHECK-NEXT:       .quad   LBB0_5
; CHECK-NEXT:       .quad   LBB0_5
; CHECK-NEXT:       .quad   LBB0_3
; CHECK-NEXT:       .size   BPF.JT.0.0, 240
