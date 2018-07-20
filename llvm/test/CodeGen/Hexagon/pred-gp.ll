; RUN: llc -march=hexagon -mcpu=hexagonv4 < %s | FileCheck %s
; Check that we are able to predicate instructions with gp-relative
; addressing mode.

@d = external global i32
@c = common global i32 0, align 4

; Function Attrs: nounwind
define i32 @test2(i8 zeroext %a, i8 zeroext %b) #0 {
; CHECK: if ({{!?}}p{{[0-3]+}}{{(.new)?}}) r{{[0-9]+}} = memw(##{{[cd]}})
; CHECK: if ({{!?}}p{{[0-3]+}}) r{{[0-9]+}} = memw(##{{[cd]}})
entry:
  %cmp = icmp eq i8 %a, %b
  br i1 %cmp, label %if.then, label %entry.if.end_crit_edge

entry.if.end_crit_edge:
  %.pre = load i32, i32* @c, align 4
  br label %if.end

if.then:
  %0 = load i32, i32* @d, align 4
  store i32 %0, i32* @c, align 4
  br label %if.end

if.end:
  %1 = phi i32 [ %.pre, %entry.if.end_crit_edge ], [ %0, %if.then ]
  ret i32 %1
}
