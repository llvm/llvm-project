; RUN: llc < %s -mtriple=i386-apple-darwin | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s

define i32 @test1(i32 %a, i8 %b) nounwind #0 {
; CHECK-LABEL: test1:
; CHECK:         crc32b
  %tmp = call i32 @llvm.x86.sse42.crc32.32.8(i32 %a, i8 %b)
  ret i32 %tmp
}

define i32 @test2(i32 %a, i8 %b) nounwind #1 {
; CHECK-LABEL: test2:
; CHECK:         crc32b
  %tmp = call i32 @llvm.x86.sse42.crc32.32.8(i32 %a, i8 %b)
  ret i32 %tmp
}

define i32 @test3(i32 %a, i8 %b) nounwind #2 {
; CHECK-LABEL: test3:
; CHECK:         crc32b
  %tmp = call i32 @llvm.x86.sse42.crc32.32.8(i32 %a, i8 %b)
  ret i32 %tmp
}

declare i32 @llvm.x86.sse42.crc32.32.8(i32, i8) nounwind

attributes #0 = { "target-features"="+crc32" }
attributes #1 = { "target-features"="+cx8,+fxsr,-3dnow,-3dnowa,-aes,-avx,-avx2,-avx512bf16,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vp2intersect,-avx512vpopcntdq,-avxvnni,-f16c,-fma,-fma4,-gfni,-kl,-mmx,-pclmul,-sha,-sse,-sse2,-sse3,-sse4.1,-sse4.2,-sse4a,-ssse3,-vaes,-vpclmulqdq,-widekl,-x87,-xop,+crc32" }
attributes #2 = { "target-features"="+crc32,+cx8,+fxsr,-3dnow,-3dnowa,-aes,-avx,-avx2,-avx512bf16,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vp2intersect,-avx512vpopcntdq,-avxvnni,-f16c,-fma,-fma4,-gfni,-kl,-mmx,-pclmul,-sha,-sse,-sse2,-sse3,-sse4.1,-sse4.2,-sse4a,-ssse3,-vaes,-vpclmulqdq,-widekl,-x87,-xop" }
