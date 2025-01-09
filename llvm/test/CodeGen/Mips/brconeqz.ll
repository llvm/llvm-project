; RUN: llc  -mtriple=mipsel-elf -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

@i = global i32 5, align 4
@result = global i32 0, align 4

define void @test() nounwind {
entry:
  %0 = load i32, ptr @i, align 4
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.end, label %if.then
; 16:	beqz	${{[0-9]+}}, $[[LABEL:[0-9A-Ba-b_]+]]
; 16: $[[LABEL]]:
if.then:                                          ; preds = %entry
  store i32 1, ptr @result, align 4
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  ret void
}

