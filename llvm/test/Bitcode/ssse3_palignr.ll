; RUN: opt < %s -S | FileCheck %s
; CHECK-NOT: {@llvm\\.palign}

define <4 x i32> @align1(<4 x i32> %a, <4 x i32> %b) nounwind readnone ssp {
entry:
  %0 = bitcast <4 x i32> %b to <2 x i64>          ; <<2 x i64>> [#uses=1]
  %1 = bitcast <4 x i32> %a to <2 x i64>          ; <<2 x i64>> [#uses=1]
  %2 = tail call <2 x i64> @llvm.x86.ssse3.palign.r.128(<2 x i64> %1, <2 x i64> %0, i8 15) ; <<2 x i64>> [#uses=1]
  %3 = bitcast <2 x i64> %2 to <4 x i32>          ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %3
}

define double @align8(<2 x i32> %a, <2 x i32> %b) nounwind readnone ssp {
entry:
  %0 = bitcast <2 x i32> %b to <1 x i64>          ; <<1 x i64>> [#uses=1]
  %1 = bitcast <2 x i32> %a to <1 x i64>          ; <<1 x i64>> [#uses=1]
  %2 = tail call <1 x i64> @llvm.x86.ssse3.palign.r(<1 x i64> %1, <1 x i64> %0, i8 7) ; <<1 x i64>> [#uses=1]
  %3 = extractelement <1 x i64> %2, i32 0         ; <i64> [#uses=1]
  %retval12 = bitcast i64 %3 to double            ; <double> [#uses=1]
  ret double %retval12
}

declare <1 x i64> @llvm.x86.ssse3.palign.r(<1 x i64>, <1 x i64>, i8) nounwind readnone

define double @align7(<2 x i32> %a, <2 x i32> %b) nounwind readnone ssp {
entry:
  %0 = bitcast <2 x i32> %b to <1 x i64>          ; <<1 x i64>> [#uses=1]
  %1 = bitcast <2 x i32> %a to <1 x i64>          ; <<1 x i64>> [#uses=1]
  %2 = tail call <1 x i64> @llvm.x86.ssse3.palign.r(<1 x i64> %1, <1 x i64> %0, i8 16) ; <<1 x i64>> [#uses=1]
  %3 = extractelement <1 x i64> %2, i32 0         ; <i64> [#uses=1]
  %retval12 = bitcast i64 %3 to double            ; <double> [#uses=1]
  ret double %retval12
}

define double @align6(<2 x i32> %a, <2 x i32> %b) nounwind readnone ssp {
entry:
  %0 = bitcast <2 x i32> %b to <1 x i64>          ; <<1 x i64>> [#uses=1]
  %1 = bitcast <2 x i32> %a to <1 x i64>          ; <<1 x i64>> [#uses=1]
  %2 = tail call <1 x i64> @llvm.x86.ssse3.palign.r(<1 x i64> %1, <1 x i64> %0, i8 9) ; <<1 x i64>> [#uses=1]
  %3 = extractelement <1 x i64> %2, i32 0         ; <i64> [#uses=1]
  %retval12 = bitcast i64 %3 to double            ; <double> [#uses=1]
  ret double %retval12
}

define double @align5(<2 x i32> %a, <2 x i32> %b) nounwind readnone ssp {
entry:
  %0 = bitcast <2 x i32> %b to <1 x i64>          ; <<1 x i64>> [#uses=1]
  %1 = bitcast <2 x i32> %a to <1 x i64>          ; <<1 x i64>> [#uses=1]
  %2 = tail call <1 x i64> @llvm.x86.ssse3.palign.r(<1 x i64> %1, <1 x i64> %0, i8 8) ; <<1 x i64>> [#uses=1]
  %3 = extractelement <1 x i64> %2, i32 0         ; <i64> [#uses=1]
  %retval12 = bitcast i64 %3 to double            ; <double> [#uses=1]
  ret double %retval12
}

define <4 x i32> @align4(<4 x i32> %a, <4 x i32> %b) nounwind readnone ssp {
entry:
  %0 = bitcast <4 x i32> %b to <2 x i64>          ; <<2 x i64>> [#uses=1]
  %1 = bitcast <4 x i32> %a to <2 x i64>          ; <<2 x i64>> [#uses=1]
  %2 = tail call <2 x i64> @llvm.x86.ssse3.palign.r.128(<2 x i64> %1, <2 x i64> %0, i8 32) ; <<2 x i64>> [#uses=1]
  %3 = bitcast <2 x i64> %2 to <4 x i32>          ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %3
}

declare <2 x i64> @llvm.x86.ssse3.palign.r.128(<2 x i64>, <2 x i64>, i8) nounwind readnone

define <4 x i32> @align3(<4 x i32> %a, <4 x i32> %b) nounwind readnone ssp {
entry:
  %0 = bitcast <4 x i32> %b to <2 x i64>          ; <<2 x i64>> [#uses=1]
  %1 = bitcast <4 x i32> %a to <2 x i64>          ; <<2 x i64>> [#uses=1]
  %2 = tail call <2 x i64> @llvm.x86.ssse3.palign.r.128(<2 x i64> %1, <2 x i64> %0, i8 17) ; <<2 x i64>> [#uses=1]
  %3 = bitcast <2 x i64> %2 to <4 x i32>          ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %3
}

define <4 x i32> @align2(<4 x i32> %a, <4 x i32> %b) nounwind readnone ssp {
entry:
  %0 = bitcast <4 x i32> %b to <2 x i64>          ; <<2 x i64>> [#uses=1]
  %1 = bitcast <4 x i32> %a to <2 x i64>          ; <<2 x i64>> [#uses=1]
  %2 = tail call <2 x i64> @llvm.x86.ssse3.palign.r.128(<2 x i64> %1, <2 x i64> %0, i8 16) ; <<2 x i64>> [#uses=1]
  %3 = bitcast <2 x i64> %2 to <4 x i32>          ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %3
}
