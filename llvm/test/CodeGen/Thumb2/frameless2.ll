; RUN: llc < %s -mtriple=thumbv7-apple-darwin -disable-fp-elim | not grep r7

%struct.noise3 = type { [3 x [17 x i32]] }
%struct.noiseguard = type { i32, i32, i32 }

define void @vorbis_encode_noisebias_setup(i8* nocapture %vi.0.7.val, double %s, i32 %block, i32* nocapture %suppress, %struct.noise3* nocapture %in, %struct.noiseguard* nocapture %guard, double %userbias) nounwind {
entry:
  %0 = getelementptr %struct.noiseguard* %guard, i32 %block, i32 2; <i32*> [#uses=1]
  %1 = load i32* %0, align 4                      ; <i32> [#uses=1]
  store i32 %1, i32* undef, align 4
  unreachable
}
