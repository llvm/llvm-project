; RUN: opt %s -disable-verify -disable-output -passes='default<O2>' -debug-pass-manager -debug-pass-manager-verbose -cgscc-npm-no-fp-rerun=1 \
; RUN:  2>&1 | FileCheck %s -check-prefixes=CHECK,NOREPS
; RUN: opt %s -disable-verify -disable-output -passes='default<O2>' -debug-pass-manager -debug-pass-manager-verbose -cgscc-npm-no-fp-rerun=0 \
; RUN:  2>&1 | FileCheck %s -check-prefixes=CHECK,REPS

; Pre-attribute the functions to avoid the PostOrderFunctionAttrsPass cause
; changes (and keep the test simple)
attributes #0 = { nofree noreturn nosync nounwind readnone }

define void @f1(void()* %p) {
  call void %p()
  ret void
}

define void @f2() #0 {
  call void @f1(void()* @f2)
  call void @f3()
  ret void
}

define void @f3() #0 {
  call void @f2()
  ret void
}

; CHECK:          Running pass: PassManager{{.*}}CGSCC
; CHECK-NEXT:     Running pass: InlinerPass on (f1)
; NOREPS:         Running analysis: FunctionStatusAnalysis on f1

; CHECK:          Running pass: PassManager{{.*}}CGSCC
; CHECK-NEXT:     Running pass: InlinerPass on (f2, f3)
; NOREPS:         Running analysis: FunctionStatusAnalysis on f2

; CHECK:          Running pass: PassManager{{.*}}CGSCC
; CHECK-NEXT:     Running pass: InlinerPass on (f2)
; REPS:           Running pass: SROA on f2
; NOREPS-NOT:     Running pass: SROA on f2

; CHECK:          Running pass: PassManager{{.*}}CGSCC
; CHECK-NEXT:     Running pass: InlinerPass on (f3)
; CHECK:          Running pass: SROA on f3
