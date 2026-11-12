; RUN: opt < %s -passes=instsimplify | llvm-dis
; PR4848
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

%0 = type { %struct.anon }
%1 = type { %0, %2, [24 x i8] }
%2 = type <{ %3, %3 }>
%3 = type { ptr, i32, %struct.rb_root, ptr, %struct.pgprot, ptr, [16 x i8] }
%struct.anon = type { }
%struct.hrtimer_clock_base = type { ptr, i32, %struct.rb_root, ptr, %struct.pgprot, ptr, %struct.pgprot, %struct.pgprot }
%struct.hrtimer_cpu_base = type { %0, [2 x %struct.hrtimer_clock_base], %struct.pgprot, i32, i64 }
%struct.pgprot = type { i64 }
%struct.rb_node = type { i64, ptr, ptr }
%struct.rb_root = type { ptr }

@per_cpu__hrtimer_bases = external global %1, align 8 ; <ptr> [#uses=1]

define void @init_hrtimers_cpu(i32 %cpu) nounwind noredzone section ".cpuinit.text" {
entry:
  %tmp3 = getelementptr %struct.hrtimer_cpu_base, ptr @per_cpu__hrtimer_bases, i32 0, i32 0 ; <ptr> [#uses=1]
  unreachable
}
