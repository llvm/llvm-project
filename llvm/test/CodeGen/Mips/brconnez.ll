; RUN: llc  -march=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

@j = global i32 0, align 4
@result = global i32 0, align 4

define void @test() nounwind {
entry:
  %0 = load i32, ptr @j, align 4
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.end, !prof !1

; 16:	bnez	${{[0-9]+}}, $[[LABEL:[0-9A-Ba-b_]+]]
; 16:   lw ${{[0-9]+}}, %got(result)(${{[0-9]+}})
; 16: $[[LABEL]]:

if.then:                                          ; preds = %entry
  store i32 1, ptr @result, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

!1 = !{!"branch_weights", i32 2, i32 1}
