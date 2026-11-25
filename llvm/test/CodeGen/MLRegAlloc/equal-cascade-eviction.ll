; REQUIRES: x86_64-linux
; RUN: rm -rf %t.rundir
; RUN: rm -rf %t.channel-basename.*
; RUN: mkdir %t.rundir
; RUN: cp %S/../../../lib/Analysis/models/log_reader.py %t.rundir
; RUN: cp %S/../../../lib/Analysis/models/interactive_host.py %t.rundir
; RUN: cp %S/Inputs/interactive_main.py %t.rundir
; RUN: %python %t.rundir/interactive_main.py %t.channel-basename \
; RUN:    llc -mtriple=x86_64-linux-unknown -regalloc=greedy -regalloc-enable-advisor=release \
; RUN:    -regalloc-evict-interactive-channel-base=%t.channel-basename %s -o /dev/null

;; A spillable range must not evict an unspillable one. The evicted range would
;; take the evictor cascade and on requeue find only ranges of that cascade.

define void @f(ptr %p, i64 %a0, i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i1 %c, i64 %l0) {
  %l01 = load volatile i64, ptr null, align 4
  %l11 = load i64, ptr null, align 4
  %l15 = load volatile i64, ptr null, align 4
  %m19 = mul i64 %a1, %a0
  store volatile i64 %m19, ptr null, align 4
  store i64 %a5, ptr null, align 4
  store i64 0, ptr %p, align 4
  %m12 = mul i64 %l11, %a5
  store volatile i64 %m12, ptr null, align 4
  %m10 = mul i64 %a0, %a2
  store i64 %m10, ptr null, align 4
  %m29 = mul i64 %a0, %a0
  store volatile i64 %m29, ptr null, align 4
  %m5 = mul i64 %a1, %a1
  %m14 = mul i64 %m5, %a0
  store volatile i64 %m14, ptr null, align 4
  store i64 %a4, ptr null, align 4
  store volatile i64 %a1, ptr null, align 4
  store volatile i64 %m5, ptr null, align 4
  store i64 %a3, ptr null, align 4
  store volatile i64 %a0, ptr null, align 4
  %m9 = mul i64 %l0, %a0
  store volatile i64 %m9, ptr null, align 4
  store i64 %a2, ptr null, align 4
  %zs27 = zext i1 %c to i8
  store volatile i8 %zs27, ptr null, align 1
  %i6 = icmp slt i64 0, %l01
  %zi6 = zext i1 %i6 to i8
  store i8 %zi6, ptr null, align 1
  ret void
}
