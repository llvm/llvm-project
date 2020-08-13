; RUN: llc  -march=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

@i = global i32 5, align 4
@j = global i32 5, align 4
@result = global i32 0, align 4

define void @test() nounwind {
entry:
  %0 = load i32, i32* @j, align 4
  %1 = load i32, i32* @i, align 4
  %cmp = icmp eq i32 %0, %1
  br i1 %cmp, label %if.then, label %if.end
; 16:	cmp	${{[0-9]+}}, ${{[0-9]+}}
; 16:	btnez	$[[LABEL:[0-9A-Ba-b_]+]]
; 16:   lw ${{[0-9]+}}, %got(result)(${{[0-9]+}})
; 16: $[[LABEL]]:

if.then:                                          ; preds = %entry
  store i32 1, i32* @result, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}


