; RUN: not llc -mtriple=loongarch64 < %s 2>&1 | FileCheck %s

;; preserve_none passes arguments in 25 GPRs and saves none of them: with all of
;; them live there is no free register for the prologue's probe loop.

; CHECK: error: {{.*}}no free register for stack probing

declare void @g(ptr)

define preserve_nonecc void @f(i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 %a6, i64 %a7, i64 %a8, i64 %a9, i64 %a10, i64 %a11, i64 %a12, i64 %a13, i64 %a14, i64 %a15, i64 %a16, i64 %a17, i64 %a18, i64 %a19, i64 %a20, i64 %a21, i64 %a22, i64 %a23, i64 %a24, i64 %a25) "probe-stack"="inline-asm" {
  %buf = alloca [40000 x i8], align 8
  %p1 = getelementptr inbounds i64, ptr %buf, i64 1
  store volatile i64 %a1, ptr %p1
  %p2 = getelementptr inbounds i64, ptr %buf, i64 2
  store volatile i64 %a2, ptr %p2
  %p3 = getelementptr inbounds i64, ptr %buf, i64 3
  store volatile i64 %a3, ptr %p3
  %p4 = getelementptr inbounds i64, ptr %buf, i64 4
  store volatile i64 %a4, ptr %p4
  %p5 = getelementptr inbounds i64, ptr %buf, i64 5
  store volatile i64 %a5, ptr %p5
  %p6 = getelementptr inbounds i64, ptr %buf, i64 6
  store volatile i64 %a6, ptr %p6
  %p7 = getelementptr inbounds i64, ptr %buf, i64 7
  store volatile i64 %a7, ptr %p7
  %p8 = getelementptr inbounds i64, ptr %buf, i64 8
  store volatile i64 %a8, ptr %p8
  %p9 = getelementptr inbounds i64, ptr %buf, i64 9
  store volatile i64 %a9, ptr %p9
  %p10 = getelementptr inbounds i64, ptr %buf, i64 10
  store volatile i64 %a10, ptr %p10
  %p11 = getelementptr inbounds i64, ptr %buf, i64 11
  store volatile i64 %a11, ptr %p11
  %p12 = getelementptr inbounds i64, ptr %buf, i64 12
  store volatile i64 %a12, ptr %p12
  %p13 = getelementptr inbounds i64, ptr %buf, i64 13
  store volatile i64 %a13, ptr %p13
  %p14 = getelementptr inbounds i64, ptr %buf, i64 14
  store volatile i64 %a14, ptr %p14
  %p15 = getelementptr inbounds i64, ptr %buf, i64 15
  store volatile i64 %a15, ptr %p15
  %p16 = getelementptr inbounds i64, ptr %buf, i64 16
  store volatile i64 %a16, ptr %p16
  %p17 = getelementptr inbounds i64, ptr %buf, i64 17
  store volatile i64 %a17, ptr %p17
  %p18 = getelementptr inbounds i64, ptr %buf, i64 18
  store volatile i64 %a18, ptr %p18
  %p19 = getelementptr inbounds i64, ptr %buf, i64 19
  store volatile i64 %a19, ptr %p19
  %p20 = getelementptr inbounds i64, ptr %buf, i64 20
  store volatile i64 %a20, ptr %p20
  %p21 = getelementptr inbounds i64, ptr %buf, i64 21
  store volatile i64 %a21, ptr %p21
  %p22 = getelementptr inbounds i64, ptr %buf, i64 22
  store volatile i64 %a22, ptr %p22
  %p23 = getelementptr inbounds i64, ptr %buf, i64 23
  store volatile i64 %a23, ptr %p23
  %p24 = getelementptr inbounds i64, ptr %buf, i64 24
  store volatile i64 %a24, ptr %p24
  %p25 = getelementptr inbounds i64, ptr %buf, i64 25
  store volatile i64 %a25, ptr %p25
  call void @g(ptr %buf)
  ret void
}
