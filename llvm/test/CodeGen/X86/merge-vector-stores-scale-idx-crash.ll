; RUN: llc < %s  -mtriple=x86_64-apple-osx10.14 -mattr=+avx2 | FileCheck %s

; Check that we don't crash due creating invalid extract_subvector indices in store merging.
; CHECK-LABEL: testfn
; CHECK: retq
define void @testfn(ptr nocapture %p) {
  %v0 = getelementptr i32, ptr %p, i64 12
  %1 = getelementptr <2 x i64>, ptr %v0, i64 -3
  store <2 x i64> undef, ptr %1, align 16
  %2 = shufflevector <4 x i64> zeroinitializer, <4 x i64> undef, <2 x i32> <i32 0, i32 1>
  %3 = getelementptr <2 x i64>, ptr %v0, i64 -2
  store <2 x i64> %2, ptr %3, align 16
  %4 = shufflevector <8 x i32> zeroinitializer, <8 x i32> undef, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  %5 = getelementptr <4 x i32>, ptr %v0, i64 -1
  store <4 x i32> %4, ptr %5, align 16
  ret void
}

%struct.o.1.5.7.9.13.15.17.19.57 = type { [0 x %struct.d.0.4.6.8.12.14.16.18.56] }
%struct.d.0.4.6.8.12.14.16.18.56 = type { float, float }

; CHECK-LABEL: testfn_scalar
; CHECK: retq
define void @testfn_scalar(ptr nocapture %j, i1 %arg) local_unnamed_addr #0 align 2 {
entry:
  %0 = bitcast i64 undef to <2 x float>
  br i1 %arg, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  unreachable

if.end:                                           ; preds = %entry
  %call.i.i17 = tail call <4 x float> @_Z1bv()
  %1 = bitcast <4 x float> %call.i.i17 to <2 x i64>
  %extract.i.i15 = extractelement <2 x i64> %1, i64 0
  %arrayidx6 = getelementptr inbounds %struct.o.1.5.7.9.13.15.17.19.57, ptr %j, i64 0, i32 0, i64 1
  store i64 %extract.i.i15, ptr %arrayidx6, align 4
  store <2 x float> %0, ptr %j, align 4
  ret void
}

declare <4 x float> @_Z1bv() local_unnamed_addr 

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="128" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

