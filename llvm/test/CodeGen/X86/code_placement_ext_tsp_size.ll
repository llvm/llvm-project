; RUN: llc -mcpu=corei7 -mtriple=x86_64-linux -apply-ext-tsp-for-size=true  < %s | FileCheck %s -check-prefix=CHECK-PERF
; RUN: llc -mcpu=corei7 -mtriple=x86_64-linux -apply-ext-tsp-for-size=false < %s | FileCheck %s -check-prefix=CHECK-SIZE

define void @func1() minsize {
;
; +-----+
; | b0  | -+
; +-----+  |
;   |      |
;   | 10   |
;   v      |
; +-----+  |
; | b1  |  | 10000
; +-----+  |
;   |      |
;   | 10   |
;   v      |
; +-----+  |
; | b2  | <+
; +-----+
;
; CHECK-PERF-LABEL: func1:
; CHECK-PERF: %b0
; CHECK-PERF: %b1
; CHECK-PERF: %b2
;
; CHECK-SIZE-LABEL: func1:
; CHECK-SIZE: %b0
; CHECK-SIZE: %b2
; CHECK-SIZE: %b1

b0:
  %call = call zeroext i1 @a()
  br i1 %call, label %b1, label %b2, !prof !1

b1:
  call void @d()
  call void @d()
  call void @d()
  br label %b2

b2:
  call void @e()
  ret void
}

define void @func_loop() minsize !prof !9 {
; Test that the algorithm can rotate loops in the presence of profile data.
;
;                  +--------+
;                  | entry  |
;                  +--------+
;                    |
;                    | 1
;                    v
; +--------+  16   +--------+
; | if.then| <---- | header | <+
; +--------+       +--------+  |
;   |                |         |
;   |                | 160     |
;   |                v         |
;   |              +--------+  |
;   |              | if.else|  | 175
;   |              +--------+  |
;   |                |         |
;   |                | 160     |
;   |                v         |
;   |        16    +--------+  |
;   +------------> | if.end | -+
;                  +--------+
;                    |
;                    | 1
;                    v
;                  +--------+
;                  |  end   |
;                  +--------+
;
; CHECK-PERF-LABEL: func_loop:
; CHECK-PERF: %entry
; CHECK-PERF: %header
; CHECK-PERF: %if.then
; CHECK-PERF: %if.else
; CHECK-PERF: %if.end
; CHECK-PERF: %end
;
; CHECK-SIZE-LABEL: func_loop:
; CHECK-SIZE: %entry
; CHECK-SIZE: %header
; CHECK-SIZE: %if.else
; CHECK-SIZE: %if.end
; CHECK-SIZE: %if.then
; CHECK-SIZE: %end

entry:
  br label %header

header:
  call void @e()
  %call = call zeroext i1 @a()
  br i1 %call, label %if.then, label %if.else, !prof !10

if.then:
  call void @f()
  br label %if.end

if.else:
  call void @g()
  br label %if.end

if.end:
  call void @h()
  %call2 = call zeroext i1 @a()
  br i1 %call2, label %header, label %end

end:
  ret void
}


declare zeroext i1 @a()
declare void @b()
declare void @c()
declare void @d()
declare void @e()
declare void @g()
declare void @f()
declare void @h()

!1 = !{!"branch_weights", i32 10, i32 10000}
!9 = !{!"function_entry_count", i64 1}
!10 = !{!"branch_weights", i32 16, i32 160}
