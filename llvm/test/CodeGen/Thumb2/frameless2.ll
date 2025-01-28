; RUN: llc < %s -mtriple=thumbv7-apple-darwin -frame-pointer=none | not grep r7

%struct.noise3 = type { [3 x [17 x i32]] }
%struct.noiseguard = type { i32, i32, i32 }

define void @vorbis_encode_noisebias_setup(ptr nocapture %vi.0.7.val, double %s, i32 %block, ptr nocapture %suppress, ptr nocapture %in, ptr nocapture %guard, double %userbias) nounwind {
entry:
  %0 = getelementptr %struct.noiseguard, ptr %guard, i32 %block, i32 2; <ptr> [#uses=1]
  %1 = load i32, ptr %0, align 4                      ; <i32> [#uses=1]
  store i32 %1, ptr undef, align 4
  unreachable
}
