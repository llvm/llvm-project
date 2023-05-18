; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -mattr=+soft-float -mips16-hard-float -relocation-model=static -O0    < %s | FileCheck %s -check-prefix=lcb

@i = global i32 0, align 4
@j = common global i32 0, align 4
@k = common global i32 0, align 4

; Function Attrs: nounwind
define i32 @s() #0 {
entry:
  %0 = load i32, ptr @i, align 4
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  store i32 0, ptr @i, align 4
  call void asm sideeffect ".space 1000", ""() #1, !srcloc !1
  br label %if.end

if.else:                                          ; preds = %entry
  store i32 1, ptr @i, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret i32 0
; lcb:	bnez	$2, $BB0_2
; lcb:	b	$BB0_1 # 16 bit inst
; lcb: $BB0_1:                                 # %if.then
}

; Function Attrs: nounwind
define i32 @b() #0 {
entry:
  %0 = load i32, ptr @i, align 4
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  store i32 0, ptr @i, align 4
  call void asm sideeffect ".space 1000000", ""() #1, !srcloc !2
  br label %if.end

if.else:                                          ; preds = %entry
  store i32 1, ptr @i, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret i32 0
}

; lcb:	beqz	$2, $BB1_1  # 16 bit inst
; lcb:	jal	$BB1_2	# branch
; lcb: $BB1_1:                                 # %if.then

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }


!1 = !{i32 65}
!2 = !{i32 167}
