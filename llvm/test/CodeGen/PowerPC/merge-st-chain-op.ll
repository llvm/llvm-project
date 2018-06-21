; RUN: llc < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@_ZNSs4_Rep20_S_empty_rep_storageE = external global [0 x i64], align 8

; Function Attrs: nounwind
define void @_ZN5clang7tooling15RefactoringTool10runAndSaveEPNS0_21FrontendActionFactoryE() #0 align 2 {
entry:
  br i1 undef, label %_ZN4llvm18IntrusiveRefCntPtrIN5clang13DiagnosticIDsEEC2EPS2_.exit, label %return

; CHECK: @_ZN5clang7tooling15RefactoringTool10runAndSaveEPNS0_21FrontendActionFactoryE

_ZN4llvm18IntrusiveRefCntPtrIN5clang13DiagnosticIDsEEC2EPS2_.exit: ; preds = %entry
  %call2 = call noalias i8* @_Znwm() #3
  %ref_cnt.i.i = bitcast i8* %call2 to i32*
  store <2 x i8*> <i8* bitcast (i64* getelementptr inbounds ([0 x i64], [0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE, i64 0, i64 3) to i8*), i8* bitcast (i64* getelementptr inbounds ([0 x i64], [0 x i64]* @_ZNSs4_Rep20_S_empty_rep_storageE, i64 0, i64 3) to i8*)>, <2 x i8*>* undef, align 8
  %IgnoreWarnings.i = getelementptr inbounds i8, i8* %call2, i64 4
  %0 = bitcast i8* %IgnoreWarnings.i to i32*
  call void @llvm.memset.p0i8.i64(i8* null, i8 0, i64 48, i32 8, i1 false) #4
  store i32 251658240, i32* %0, align 4
  store i256 37662610426935100959726589394453639584271499769928088551424, i256* null, align 8
  store i32 1, i32* %ref_cnt.i.i, align 4
  unreachable

return:                                           ; preds = %entry
  ret void
}

; Function Attrs: nobuiltin
declare noalias i8* @_Znwm() #1

; Function Attrs: nounwind argmemonly
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) #2

attributes #0 = { nounwind "target-cpu"="pwr7" }
attributes #1 = { nobuiltin "target-cpu"="pwr7" }
attributes #2 = { nounwind argmemonly }
attributes #3 = { builtin nounwind }
attributes #4 = { nounwind }

