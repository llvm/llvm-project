; RUN: llc -mtriple=mipsel-linux-gnu -mattr=mips16 -mattr=+soft-float -mips16-hard-float -relocation-model=static -mips16-constant-islands < %s | FileCheck %s -check-prefix=constisle

@i = common global i32 0, align 4
@b = common global i32 0, align 4
@l = common global i32 0, align 4

; Function Attrs: nounwind
define void @foo() #0 {
entry:
  store i32 305419896, ptr @i, align 4
  %0 = load i32, ptr @b, align 4
  %tobool = icmp ne i32 %0, 0
  br i1 %tobool, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  store i32 10, ptr @b, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  store i32 20, ptr @b, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  call void asm sideeffect ".space 100000", ""() #1, !srcloc !1
  store i32 305419896, ptr @l, align 4
  ret void
; constisle: $CPI0_1:
; constisle	.4byte	305419896               # 0x12345678
; constisle	#APP
; constisle	.space 100000
; constisle	#NO_APP
; constisle $CPI0_0:
; constisle	.4byte	305419896               # 0x12345678
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!1 = !{i32 103}
