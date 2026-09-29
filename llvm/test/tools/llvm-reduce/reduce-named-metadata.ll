; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=named-metadata --test=FileCheck --test-arg=--check-prefix=CHECK-INTERESTINGNESS --test-arg=%s --test-arg=--input-file %s -o %t
; RUN: FileCheck --check-prefix=RESULT %s < %t
; Test the various named metadata recognized for simple list behavior.

; CHECK-INTERESTINGNESS: !llvm.ident = !{![[LLVM_IDENT:[0-9]+]]
; CHECK-INTERESTINGNESS: !opencl.spir.version = !{{{.*}}![[SPIR_VERSION:[0-9]+]]}
; CHECK-INTERESTINGNESS: !opencl.ocl.version = !{{{.*}}![[OCL_VERSION:[0-9]+]]}
; CHECK-INTERESTINGNESS: !opencl.used.extensions = !{{{.*}}![[OCL_EXTENSION:[0-9]+]]}
; CHECK-INTERESTINGNESS: !opencl.used.optional.core.features = !{{{.*}}![[OCL_OPTIONAL_CORE_FEATURE:[0-9]+]]}
; CHECK-INTERESTINGNESS: !opencl.compiler.options = !{{{.*}}![[OCL_COMPILER_OPTIONS:[0-9]+]]}

; CHECK-DAG: CHECK-INTERESTINGNESS: ![[LLVM_IDENT]] = !{!"some llvm version 0"}
; CHECK-DAG: CHECK-INTERESTINGNESS: ![[SPIR_VERSION]] = !{!"some spir version 1"}
; CHECK-DAG: CHECK-INTERESTINGNESS: ![[OCL_VERSION]] = !{!"some ocl version 1"}
; CHECK-DAG: CHECK-INTERESTINGNESS: ![[OCL_EXTENSION]] = !{!"some ocl extension 1"}
; CHECK-DAG: CHECK-INTERESTINGNESS: ![[OCL_OPTIONAL_CORE_FEATURE]] = !{!"some ocl optional core feature 1"}
; CHECK-DAG: CHECK-INTERESTINGNESS: ![[OCL_COMPILER_OPTIONS]] = !{!"some ocl compiler option 1"}


; RESULT: !llvm.ident = !{![[LLVM_IDENT:[0-9]+]]
; RESULT: !opencl.spir.version = !{![[SPIR_VERSION:[0-9]+]]}
; RESULT: !opencl.ocl.version = !{![[OCL_VERSION:[0-9]+]]}
; RESULT: !opencl.used.extensions = !{![[OCL_EXTENSION:[0-9]+]]}
; RESULT: !opencl.used.optional.core.features = !{![[OCL_OPTIONAL_CORE_FEATURE:[0-9]+]]}
; RESULT: !opencl.compiler.options = !{![[OCL_COMPILER_OPTION:[0-9]+]]}
; RESULT: !some.unknown.named = !{![[UNKNOWN_0:[0-9]+]], ![[UNKNOWN_1:[0-9]+]]}


; RESULT: ![[LLVM_IDENT]] = !{!"some llvm version 0"}
; RESULT: ![[SPIR_VERSION]] = !{!"some spir version 1"}
; RESULT: ![[OCL_VERSION]] = !{!"some ocl version 1"}
; RESULT: ![[OCL_EXTENSION]] = !{!"some ocl extension 1"}
; RESULT: ![[OCL_OPTIONAL_CORE_FEATURE]] = !{!"some ocl optional core feature 1"}
; RESULT: ![[OCL_COMPILER_OPTION]] = !{!"some ocl compiler option 1"}
; RESULT: ![[UNKNOWN_0]] = !{!"some unknown option 0"}
; RESULT: ![[UNKNOWN_1]] = !{!"some unknown option 1"}

!llvm.ident = !{!0, !1, !0}
!opencl.spir.version = !{!2, !3}
!opencl.ocl.version = !{!4, !5}
!opencl.used.extensions = !{!6, !7}
!opencl.used.optional.core.features = !{!8, !9}
!opencl.compiler.options = !{!10, !11}
!some.unknown.named = !{!12, !13}

!0 = !{!"some llvm version 0"}
!1 = !{!"some llvm version 1"}
!2 = !{!"some spir version 0"}
!3 = !{!"some spir version 1"}
!4 = !{!"some ocl version 0"}
!5 = !{!"some ocl version 1"}
!6 = !{!"some ocl extension 0"}
!7 = !{!"some ocl extension 1"}
!8 = !{!"some ocl optional core feature 0"}
!9 = !{!"some ocl optional core feature 1"}
!10 = !{!"some ocl compiler option 0"}
!11 = !{!"some ocl compiler option 1"}
!12 = !{!"some unknown option 0"}
!13 = !{!"some unknown option 1"}
