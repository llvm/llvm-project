; RUN: sed 's/EVL_ARG/i32 5/g' %s | not llubi --verbose 2>&1 | FileCheck %s --check-prefix=TOO-LARGE
; RUN: sed 's/EVL_ARG/i32 poison/g' %s | not llubi --verbose 2>&1 | FileCheck %s --check-prefix=POISON

define void @main() {
  %res = call <4 x i32> @llvm.vp.add.v4i32(<4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <4 x i1> splat (i1 true), EVL_ARG)
  ret void
}

; TOO-LARGE: Immediate UB detected: VP intrinsic explicit vector length exceeds the runtime vector length. EVL: 5, Vector length: 4.
; POISON: Immediate UB detected: VP intrinsic with poison explicit vector length.
