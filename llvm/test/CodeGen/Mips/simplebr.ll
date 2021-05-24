; RUN: llc -mtriple=mipsel-linux-gnu -march=mipsel -mattr=mips16 -mips16-hard-float -mattr=+soft-float -relocation-model=static < %s | FileCheck %s -check-prefix=CHECK-STATIC16

; ModuleID = 'simplebr.c'
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:32-i16:16:32-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32-S64"
target triple = "mips--linux-gnu"

@i = common global i32 0, align 4

; Function Attrs: nounwind
define void @foo() #0 {
entry:
  %0 = load i32, i32* @i, align 4
  %tobool = icmp ne i32 %0, 0
  br i1 %tobool, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  call void bitcast (void (...)* @goo to void ()*)()
  br label %if.end

if.else:                                          ; preds = %entry
  call void bitcast (void (...)* @hoo to void ()*)()
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

; CHECK-STATIC16:	b	$BB{{[0-9]+}}_{{[0-9]+}} # 16 bit inst

declare void @goo(...) #1

declare void @hoo(...) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="true" }
attributes #1 = { "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="true" }


