; RUN: llc  -march=mipsel -mcpu=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

@i = global i32 -5, align 4
@j = global i32 10, align 4
@k = global i32 -5, align 4
@result1 = global i32 0, align 4
@result2 = global i32 1, align 4

define void @test() nounwind {
entry:
  %0 = load i32* @j, align 4
  %1 = load i32* @i, align 4
  %cmp = icmp sgt i32 %0, %1
  br i1 %cmp, label %if.then, label %if.end

; 16:	slt	${{[0-9]+}}, ${{[0-9]+}}
; 16:	bteqz	$[[LABEL:[0-9A-Ba-b_]+]]
; 16: $[[LABEL]]:

if.then:                                          ; preds = %entry
  store i32 1, i32* @result1, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %2 = load i32* @k, align 4
  %cmp1 = icmp sgt i32 %1, %2
  br i1 %cmp1, label %if.then2, label %if.end3

if.then2:                                         ; preds = %if.end
  store i32 0, i32* @result1, align 4
  br label %if.end3

if.end3:                                          ; preds = %if.then2, %if.end
  ret void
}


