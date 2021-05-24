; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -mattr=+soft-float -mips16-hard-float -relocation-model=static -mips16-constant-islands   < %s | FileCheck %s -check-prefix=jal16

@j = global i32 10, align 4
@.str = private unnamed_addr constant [11 x i8] c"at bottom\0A\00", align 1
@i = common global i32 0, align 4

; Function Attrs: nounwind
define i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  br label %z

z:                                                ; preds = %y, %entry
  %call = call i32 bitcast (i32 (...)* @foo to i32 ()*)()
  call void asm sideeffect ".space 10000000", ""() #2, !srcloc !1
  br label %y

y:                                                ; preds = %z
  %call1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i32 0, i32 0))
  br label %z

return:                                           ; No predecessors!
  %0 = load i32, i32* %retval
  ret i32 %0
; jal16: 	jal	$BB{{[0-9]+}}_{{[0-9]+}}
}

declare i32 @foo(...) #1

declare i32 @printf(i8*, ...) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!1 = !{i32 68}
