; RUN: llc -march=hexagon -mcpu=hexagonv4 < %s | FileCheck %s
; Check that gp-relative instructions are being generated.

@a = common global i32 0, align 4
@b = common global i32 0, align 4
@c = common global i32 0, align 4

define i32 @foo(i32 %p) #0 {
entry:
; CHECK: r{{[0-9]+}} = memw(gp+#a)
; CHECK: r{{[0-9]+}} = memw(gp+#b)
; CHECK: if (p{{[0-3]}}) memw(##c) = r{{[0-9]+}}
  %0 = load i32, i32* @a, align 4
  %1 = load i32, i32* @b, align 4
  %add = add nsw i32 %1, %0
  %cmp = icmp eq i32 %0, %1
  br i1 %cmp, label %if.then, label %entry.if.end_crit_edge

entry.if.end_crit_edge:
  %.pre = load i32, i32* @c, align 4
  br label %if.end

if.then:
  %add1 = add nsw i32 %add, %0
  store i32 %add1, i32* @c, align 4
  br label %if.end

if.end:
  %2 = phi i32 [ %.pre, %entry.if.end_crit_edge ], [ %add1, %if.then ]
  %cmp2 = icmp eq i32 %add, %2
  %sel1 = select i1 %cmp2, i32 %2, i32 %1
  ret i32 %sel1
}
