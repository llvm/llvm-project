; REQUIRES: x86-registered-target

; The "long-double-type" module flag selects the long double format of
; the runtime library calls.

; RUN: split-file %s %t

; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=x86_64-unknown-linux-gnu < %t/fp128.ll | FileCheck --check-prefix=FP128 %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=x86_64-unknown-linux-gnu < %t/double.ll | FileCheck --check-prefix=DOUBLE %s

; No module flag: the format defaults to the one implied by the target triple.
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=x86_64-unknown-linux-gnu < %t/none.ll | FileCheck --check-prefix=X86FP80 %s

;--- fp128.ll
; FP128: declare double @nexttoward(double, fp128)
; FP128: declare fp128 @sinl(fp128)
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"long-double-type", !"fp128"}

;--- double.ll
; DOUBLE: declare double @nexttoward(double, double)
; DOUBLE-NOT: @sinl(
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"long-double-type", !"double"}

;--- none.ll
; X86FP80: declare double @nexttoward(double, x86_fp80)
; X86FP80: declare x86_fp80 @sinl(x86_fp80)
