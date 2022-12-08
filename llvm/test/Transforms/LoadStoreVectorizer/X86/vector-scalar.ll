; RUN: opt -mtriple=x86_64-unknown-linux-gnu -passes=load-store-vectorizer -mcpu haswell -S -o - %s | FileCheck %s
; RUN: opt -mtriple=x86_64-unknown-linux-gnu -aa-pipeline=basic-aa -passes='function(load-store-vectorizer)' -mcpu haswell -S -o - %s | FileCheck %s

; Check that the LoadStoreVectorizer does not crash due to not differentiating <1 x T> and T.

; CHECK-LABEL: @vector_scalar(
; CHECK: store double
; CHECK: store <1 x double>
define void @vector_scalar(double* %ptr, double %a, <1 x double> %b) {
  %1 = bitcast double* %ptr to <1 x double>*
  %2 = getelementptr <1 x double>, <1 x double>* %1, i32 1
  store double %a, double* %ptr, align 8
  store <1 x double> %b, <1 x double>* %2, align 8
  ret void
}
