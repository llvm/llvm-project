; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7 | FileCheck %s

; Make sure that values of illegal types are not scalarized between basic blocks.
;CHECK: test
;CHECK-NOT: pinsrw
;CHECK-NOT: pextrb
;CHECK: ret
define void @test(i1 %cond) {
ENTRY:
  br label %LOOP
LOOP:
  %vec1 = phi <4 x i1> [ %vec1_or_2, %LOOP ], [ zeroinitializer, %ENTRY ]
  %vec2 = phi <4 x i1> [ %vec2_and_1, %LOOP ], [ zeroinitializer, %ENTRY ]
  %vec1_or_2 = or <4 x i1> %vec1, %vec2
  %vec2_and_1 = and <4 x i1> %vec2, %vec1
  br i1 %cond, label %LOOP, label %EXIT

EXIT:
  ret void
}

