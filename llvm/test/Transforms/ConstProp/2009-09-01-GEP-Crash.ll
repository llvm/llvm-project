; RUN: opt < %s -constprop | llvm-dis
; PR4848
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

%0 = type { %struct.anon }
%1 = type { %0, %2, [24 x i8] }
%2 = type <{ %3, %3 }>
%3 = type { %struct.hrtimer_cpu_base*, i32, %struct.rb_root, %struct.rb_node*, %struct.pgprot, i64 ()*, [16 x i8] }
%struct.anon = type { }
%struct.hrtimer_clock_base = type { %struct.hrtimer_cpu_base*, i32, %struct.rb_root, %struct.rb_node*, %struct.pgprot, i64 ()*, %struct.pgprot, %struct.pgprot }
%struct.hrtimer_cpu_base = type { %0, [2 x %struct.hrtimer_clock_base], %struct.pgprot, i32, i64 }
%struct.pgprot = type { i64 }
%struct.rb_node = type { i64, %struct.rb_node*, %struct.rb_node* }
%struct.rb_root = type { %struct.rb_node* }

@per_cpu__hrtimer_bases = external global %1, align 8 ; <%1*> [#uses=1]

define void @init_hrtimers_cpu(i32 %cpu) nounwind noredzone section ".cpuinit.text" {
entry:
  %tmp3 = getelementptr %struct.hrtimer_cpu_base* bitcast (%1* @per_cpu__hrtimer_bases to %struct.hrtimer_cpu_base*), i32 0, i32 0 ; <%0*> [#uses=1]
  %tmp5 = bitcast %0* %tmp3 to i8*                ; <i8*> [#uses=0]
  unreachable
}
