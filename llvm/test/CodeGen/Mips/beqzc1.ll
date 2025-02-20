; RUN: llc -mtriple=mipsel-linux-gnu -mattr=mips16 -mattr=+soft-float -mips16-hard-float -relocation-model=pic -mips16-constant-islands < %s | FileCheck %s -check-prefix=cond-b-short

@i = global i32 0, align 4
@j = common global i32 0, align 4

; Function Attrs: nounwind optsize
define i32 @main() #0 {
entry:
  %0 = load i32, ptr @i, align 4
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.end

; cond-b-short: 	bnez	${{[0-9]+}}, $BB{{[0-9]+}}_{{[0-9]+}}  # 16 bit inst
if.then:                                          ; preds = %entry
  store i32 10, ptr @j, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret i32 0
}

attributes #0 = { nounwind optsize "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="true" }


