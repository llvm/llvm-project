; RUN: llc < %s -o /dev/null

;
; Test that we dont get a crash after an invalid build_vector combine.
;

target triple = "aarch64-unknown-linux-gnu"
define void @test_crash(ptr %dst_ptr) {
entry:
  %vec_load = load <4 x i16>, ptr undef, align 8
  %0 = sext <4 x i16> %vec_load to <4 x i32>
  %add71vec = add nsw <4 x i32> %0, <i32 32, i32 32, i32 32, i32 32>
  %add104vec = add nsw <4 x i32> %add71vec, zeroinitializer
  %add105vec = add nsw <4 x i32> zeroinitializer, %add104vec
  %vec = lshr <4 x i32> %add105vec, <i32 6, i32 6, i32 6, i32 6>
  %1 = trunc <4 x i32> %vec to <4 x i16>
  %2 = shufflevector <4 x i16> %1, <4 x i16> undef, <2 x i32> <i32 1, i32 2>
  %3 = sext <2 x i16> %2 to <2 x i32>
  %4 = shufflevector <2 x i32> %3, <2 x i32> poison, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  %5 = shufflevector <4 x i32> undef, <4 x i32> %4, <4 x i32> <i32 0, i32 4, i32 5, i32 undef>
  %6 = insertelement <4 x i32> %5, i32 undef, i64 3
  %7 = add nsw <4 x i32> %6, zeroinitializer
  %8 = select <4 x i1> zeroinitializer, <4 x i32> %7, <4 x i32> undef
  %9 = trunc <4 x i32> %8 to <4 x i8>
  store <4 x i8> %9, ptr %dst_ptr, align 1
  ret void
}
