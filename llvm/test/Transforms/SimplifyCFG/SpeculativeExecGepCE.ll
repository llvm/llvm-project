; RUN: opt < %s -simplifycfg -phi-node-folding-threshold=0 -S | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

@d_buf = internal constant [8 x i8] [i8 126, i8 127, i8 128, i8 129, i8 130, i8 131, i8 132, i8 133], align 8
@a = internal constant { i8*, i64} {i8* getelementptr inbounds ([8 x i8], [8 x i8]* @d_buf, i64 0, i64 0), i64 0}

; CHECK-LABEL: @test
; CHECK-LABEL: end:
; CHECK: %x1 = phi i8*
define i8* @test(i1* %dummy, i8* %a, i8* %b, i8 %v) {

entry:
  %cond1 = load volatile i1, i1* %dummy
  br i1 %cond1, label %if, label %end

if:
  %cond2 = load volatile i1, i1* %dummy
  br i1 %cond2, label %then, label %end

then:
  br label %end

end:
  %x1 = phi i8* [ %a, %entry ], [ %b, %if ], [getelementptr inbounds ([8 x i8], [8 x i8]* @d_buf, i64 0, i64 0) , %then ]

  ret i8* %x1
}
