; RUN: llvm-as < %s | opt -gvn -dse | llvm-dis | grep {call.*memcpy} | count 1

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin8"
	%struct.ggFrame3 = type { %struct.ggPoint3, %struct.ggONB3 }
	%struct.ggHMatrix3 = type { [4 x [4 x double]] }
	%struct.ggONB3 = type { %struct.ggPoint3, %struct.ggPoint3, %struct.ggPoint3 }
	%struct.ggPoint3 = type { [3 x double] }
	%struct.ggQuaternion = type { [4 x double], i32, %struct.ggHMatrix3 }

declare void @llvm.memcpy.i64(i8*, i8*, i64, i32) nounwind 

define void @_Z10ggCRSplineRK8ggFrame3S1_S1_S1_d(%struct.ggFrame3* noalias sret  %agg.result, %struct.ggFrame3* %f0, %struct.ggFrame3* %f1, %struct.ggFrame3* %f2, %struct.ggFrame3* %f3, double %t) nounwind  {
entry:
	%qresult = alloca %struct.ggQuaternion		; <%struct.ggQuaternion*> [#uses=1]
	%tmp = alloca %struct.ggONB3		; <%struct.ggONB3*> [#uses=2]
	call void @_ZN12ggQuaternion7getONB3Ev( %struct.ggONB3* noalias sret  %tmp, %struct.ggQuaternion* %qresult ) nounwind 
	%tmp1.i = getelementptr %struct.ggFrame3* %agg.result, i32 0, i32 1		; <%struct.ggONB3*> [#uses=1]
	%tmp13.i = bitcast %struct.ggONB3* %tmp1.i to i8*		; <i8*> [#uses=1]
	%tmp24.i = bitcast %struct.ggONB3* %tmp to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i64( i8* %tmp13.i, i8* %tmp24.i, i64 72, i32 8 ) nounwind 
	ret void
}

declare void @_ZN12ggQuaternion7getONB3Ev(%struct.ggONB3* noalias sret , %struct.ggQuaternion*) nounwind 
