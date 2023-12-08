; RUN: llc < %s -mtriple=x86_64--

define void @a() local_unnamed_addr #0 {
  ret void
}

define void @b() local_unnamed_addr #1 {
  ret void
}

attributes #0 = { "target-features"="+avx,+avx2,+avx512bw,+avx512f,+f16c,+fma,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave" }
attributes #1 = { "target-features"="+avx,+avx2,+avx512f,+avx512vl,+f16c,+fma,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave" }
