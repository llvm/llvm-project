; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

;; This test checks SYCL relational builtin any and all with vector input types.

; CHECK-SPIRV: %[[#BoolTypeID:]] = OpTypeBool

; CHECK-SPIRV: OpAny %[[#BoolTypeID]]
; CHECK-SPIRV: OpAny %[[#BoolTypeID]]
; CHECK-SPIRV: OpAny %[[#BoolTypeID]]
; CHECK-SPIRV: OpAny %[[#BoolTypeID]]
; CHECK-SPIRV: OpAll %[[#BoolTypeID]]
; CHECK-SPIRV: OpAll %[[#BoolTypeID]]
; CHECK-SPIRV: OpAll %[[#BoolTypeID]]
; CHECK-SPIRV: OpAll %[[#BoolTypeID]]

define dso_local spir_func void @test_vector(i32 addrspace(4)* nocapture writeonly %out, <2 x i8> %c, <2 x i16> %s, <2 x i32> %i, <2 x i64> %l) local_unnamed_addr {
entry:
  %call = tail call spir_func i32 @_Z3anyDv2_c(<2 x i8> %c)
  %call1 = tail call spir_func i32 @_Z3anyDv2_s(<2 x i16> %s)
  %add = add nsw i32 %call1, %call
  %call2 = tail call spir_func i32 @_Z3anyDv2_i(<2 x i32> %i)
  %add3 = add nsw i32 %add, %call2
  %call4 = tail call spir_func i32 @_Z3anyDv2_l(<2 x i64> %l)
  %add5 = add nsw i32 %add3, %call4
  %call6 = tail call spir_func i32 @_Z3allDv2_c(<2 x i8> %c)
  %add7 = add nsw i32 %add5, %call6
  %call8 = tail call spir_func i32 @_Z3allDv2_s(<2 x i16> %s)
  %add9 = add nsw i32 %add7, %call8
  %call10 = tail call spir_func i32 @_Z3allDv2_i(<2 x i32> %i)
  %add11 = add nsw i32 %add9, %call10
  %call12 = tail call spir_func i32 @_Z3allDv2_l(<2 x i64> %l)
  %add13 = add nsw i32 %add11, %call12
  store i32 %add13, i32 addrspace(4)* %out, align 4
  ret void
}

declare spir_func i32 @_Z3anyDv2_c(<2 x i8>) local_unnamed_addr

declare spir_func i32 @_Z3anyDv2_s(<2 x i16>) local_unnamed_addr

declare spir_func i32 @_Z3anyDv2_i(<2 x i32>) local_unnamed_addr

declare spir_func i32 @_Z3anyDv2_l(<2 x i64>) local_unnamed_addr

declare spir_func i32 @_Z3allDv2_c(<2 x i8>) local_unnamed_addr

declare spir_func i32 @_Z3allDv2_s(<2 x i16>) local_unnamed_addr

declare spir_func i32 @_Z3allDv2_i(<2 x i32>) local_unnamed_addr

declare spir_func i32 @_Z3allDv2_l(<2 x i64>) local_unnamed_addr
