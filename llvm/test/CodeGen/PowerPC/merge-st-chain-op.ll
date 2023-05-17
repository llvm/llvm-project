; RUN: llc -verify-machineinstrs < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@_ZNSs4_Rep20_S_empty_rep_storageE = external global [0 x i64], align 8

; Function Attrs: nounwind
define void @_ZN5clang7tooling15RefactoringTool10runAndSaveEPNS0_21FrontendActionFactoryE() #0 align 2 {
entry:
  br i1 undef, label %_ZN4llvm18IntrusiveRefCntPtrIN5clang13DiagnosticIDsEEC2EPS2_.exit, label %return

; CHECK: @_ZN5clang7tooling15RefactoringTool10runAndSaveEPNS0_21FrontendActionFactoryE

_ZN4llvm18IntrusiveRefCntPtrIN5clang13DiagnosticIDsEEC2EPS2_.exit: ; preds = %entry
  %call2 = call noalias ptr @_Znwm() #3
  store <2 x ptr> <ptr getelementptr inbounds ([0 x i64], ptr @_ZNSs4_Rep20_S_empty_rep_storageE, i64 0, i64 3), ptr getelementptr inbounds ([0 x i64], ptr @_ZNSs4_Rep20_S_empty_rep_storageE, i64 0, i64 3)>, ptr undef, align 8
  %IgnoreWarnings.i = getelementptr inbounds i8, ptr %call2, i64 4
  call void @llvm.memset.p0.i64(ptr align 8 null, i8 0, i64 48, i1 false) #4
  store i32 251658240, ptr %IgnoreWarnings.i, align 4
  store i256 37662610426935100959726589394453639584271499769928088551424, ptr null, align 8
  store i32 1, ptr %call2, align 4
  unreachable

return:                                           ; preds = %entry
  ret void
}

; Function Attrs: nobuiltin
declare noalias ptr @_Znwm() #1

; Function Attrs: nounwind argmemonly
declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1) #2

attributes #0 = { nounwind "target-cpu"="pwr7" }
attributes #1 = { nobuiltin "target-cpu"="pwr7" }
attributes #2 = { nounwind argmemonly }
attributes #3 = { builtin nounwind }
attributes #4 = { nounwind }

