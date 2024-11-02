; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

;; This test checks following SYCL relational builtins with half and half2 types:
;;   isfinite, isinf, isnan, isnormal, signbit, isequal, isnotequal, isgreater
;;   isgreaterequal, isless, islessequal, islessgreater, isordered, isunordered

; CHECK-SPIRV: %[[#BoolTypeID:]] = OpTypeBool
; CHECK-SPIRV: %[[#BoolVectorTypeID:]] = OpTypeVector %[[#BoolTypeID]] 2

; CHECK-SPIRV: OpIsFinite %[[#BoolTypeID]]
; CHECK-SPIRV: OpIsInf %[[#BoolTypeID]]
; CHECK-SPIRV: OpIsNan %[[#BoolTypeID]]
; CHECK-SPIRV: OpIsNormal %[[#BoolTypeID]]
; CHECK-SPIRV: OpSignBitSet %[[#BoolTypeID]]
; CHECK-SPIRV: OpFOrdEqual %[[#BoolTypeID]]
; CHECK-SPIRV: OpFUnordNotEqual %[[#BoolTypeID]]
; CHECK-SPIRV: OpFOrdGreaterThan %[[#BoolTypeID]]
; CHECK-SPIRV: OpFOrdGreaterThanEqual %[[#BoolTypeID]]
; CHECK-SPIRV: OpFOrdLessThan %[[#BoolTypeID]]
; CHECK-SPIRV: OpFOrdLessThanEqual %[[#BoolTypeID]]
; CHECK-SPIRV: OpFOrdNotEqual %[[#BoolTypeID]]
; CHECK-SPIRV: OpOrdered %[[#BoolTypeID]]
; CHECK-SPIRV: OpUnordered %[[#BoolTypeID]]

; CHECK-SPIRV: OpIsFinite %[[#BoolVectorTypeID]]
; CHECK-SPIRV: OpIsInf %[[#BoolVectorTypeID]]
; CHECK-SPIRV: OpIsNan %[[#BoolVectorTypeID]]
; CHECK-SPIRV: OpIsNormal %[[#BoolVectorTypeID]]
; CHECK-SPIRV: OpSignBitSet %[[#BoolVectorTypeID]]
; CHECK-SPIRV: OpFOrdEqual %[[#BoolVectorTypeID]]
; CHECK-SPIRV: OpFUnordNotEqual %[[#BoolVectorTypeID]]
; CHECK-SPIRV: OpFOrdGreaterThan %[[#BoolVectorTypeID]]
; CHECK-SPIRV: OpFOrdGreaterThanEqual %[[#BoolVectorTypeID]]
; CHECK-SPIRV: OpFOrdLessThan %[[#BoolVectorTypeID]]
; CHECK-SPIRV: OpFOrdLessThanEqual %[[#BoolVectorTypeID]]
; CHECK-SPIRV: OpFOrdNotEqual %[[#BoolVectorTypeID]]
; CHECK-SPIRV: OpOrdered %[[#BoolVectorTypeID]]
; CHECK-SPIRV: OpUnordered %[[#BoolVectorTypeID]]

define dso_local spir_func void @test_scalar(i32 addrspace(4)* nocapture writeonly %out, half %h) local_unnamed_addr {
entry:
  %call = tail call spir_func i32 @_Z8isfiniteDh(half %h)
  %call1 = tail call spir_func i32 @_Z5isinfDh(half %h)
  %add = add nsw i32 %call1, %call
  %call2 = tail call spir_func i32 @_Z5isnanDh(half %h)
  %add3 = add nsw i32 %add, %call2
  %call4 = tail call spir_func i32 @_Z8isnormalDh(half %h)
  %add5 = add nsw i32 %add3, %call4
  %call6 = tail call spir_func i32 @_Z7signbitDh(half %h)
  %add7 = add nsw i32 %add5, %call6
  %call8 = tail call spir_func i32 @_Z7isequalDhDh(half %h, half %h)
  %add9 = add nsw i32 %add7, %call8
  %call10 = tail call spir_func i32 @_Z10isnotequalDhDh(half %h, half %h)
  %add11 = add nsw i32 %add9, %call10
  %call12 = tail call spir_func i32 @_Z9isgreaterDhDh(half %h, half %h)
  %add13 = add nsw i32 %add11, %call12
  %call14 = tail call spir_func i32 @_Z14isgreaterequalDhDh(half %h, half %h)
  %add15 = add nsw i32 %add13, %call14
  %call16 = tail call spir_func i32 @_Z6islessDhDh(half %h, half %h)
  %add17 = add nsw i32 %add15, %call16
  %call18 = tail call spir_func i32 @_Z11islessequalDhDh(half %h, half %h)
  %add19 = add nsw i32 %add17, %call18
  %call20 = tail call spir_func i32 @_Z13islessgreaterDhDh(half %h, half %h)
  %add21 = add nsw i32 %add19, %call20
  %call22 = tail call spir_func i32 @_Z9isorderedDhDh(half %h, half %h)
  %add23 = add nsw i32 %add21, %call22
  %call24 = tail call spir_func i32 @_Z11isunorderedDhDh(half %h, half %h)
  %add25 = add nsw i32 %add23, %call24
  store i32 %add25, i32 addrspace(4)* %out, align 4
  ret void
}

declare spir_func i32 @_Z8isfiniteDh(half) local_unnamed_addr

declare spir_func i32 @_Z5isinfDh(half) local_unnamed_addr

declare spir_func i32 @_Z5isnanDh(half) local_unnamed_addr

declare spir_func i32 @_Z8isnormalDh(half) local_unnamed_addr

declare spir_func i32 @_Z7signbitDh(half) local_unnamed_addr

declare spir_func i32 @_Z7isequalDhDh(half, half) local_unnamed_addr

declare spir_func i32 @_Z10isnotequalDhDh(half, half) local_unnamed_addr

declare spir_func i32 @_Z9isgreaterDhDh(half, half) local_unnamed_addr

declare spir_func i32 @_Z14isgreaterequalDhDh(half, half) local_unnamed_addr

declare spir_func i32 @_Z6islessDhDh(half, half) local_unnamed_addr

declare spir_func i32 @_Z11islessequalDhDh(half, half) local_unnamed_addr

declare spir_func i32 @_Z13islessgreaterDhDh(half, half) local_unnamed_addr

declare spir_func i32 @_Z9isorderedDhDh(half, half) local_unnamed_addr

declare spir_func i32 @_Z11isunorderedDhDh(half, half) local_unnamed_addr

define dso_local spir_func void @test_vector(<2 x i16> addrspace(4)* nocapture writeonly %out, <2 x half> %h) local_unnamed_addr {
entry:
  %call = tail call spir_func <2 x i16> @_Z8isfiniteDv2_Dh(<2 x half> %h)
  %call1 = tail call spir_func <2 x i16> @_Z5isinfDv2_Dh(<2 x half> %h)
  %add = add <2 x i16> %call1, %call
  %call2 = tail call spir_func <2 x i16> @_Z5isnanDv2_Dh(<2 x half> %h)
  %add3 = add <2 x i16> %add, %call2
  %call4 = tail call spir_func <2 x i16> @_Z8isnormalDv2_Dh(<2 x half> %h)
  %add5 = add <2 x i16> %add3, %call4
  %call6 = tail call spir_func <2 x i16> @_Z7signbitDv2_Dh(<2 x half> %h)
  %add7 = add <2 x i16> %add5, %call6
  %call8 = tail call spir_func <2 x i16> @_Z7isequalDv2_DhS_(<2 x half> %h, <2 x half> %h)
  %add9 = add <2 x i16> %add7, %call8
  %call10 = tail call spir_func <2 x i16> @_Z10isnotequalDv2_DhS_(<2 x half> %h, <2 x half> %h)
  %add11 = add <2 x i16> %add9, %call10
  %call12 = tail call spir_func <2 x i16> @_Z9isgreaterDv2_DhS_(<2 x half> %h, <2 x half> %h)
  %add13 = add <2 x i16> %add11, %call12
  %call14 = tail call spir_func <2 x i16> @_Z14isgreaterequalDv2_DhS_(<2 x half> %h, <2 x half> %h)
  %add15 = add <2 x i16> %add13, %call14
  %call16 = tail call spir_func <2 x i16> @_Z6islessDv2_DhS_(<2 x half> %h, <2 x half> %h)
  %add17 = add <2 x i16> %add15, %call16
  %call18 = tail call spir_func <2 x i16> @_Z11islessequalDv2_DhS_(<2 x half> %h, <2 x half> %h)
  %add19 = add <2 x i16> %add17, %call18
  %call20 = tail call spir_func <2 x i16> @_Z13islessgreaterDv2_DhS_(<2 x half> %h, <2 x half> %h)
  %add21 = add <2 x i16> %add19, %call20
  %call22 = tail call spir_func <2 x i16> @_Z9isorderedDv2_DhS_(<2 x half> %h, <2 x half> %h)
  %add23 = add <2 x i16> %add21, %call22
  %call24 = tail call spir_func <2 x i16> @_Z11isunorderedDv2_DhS_(<2 x half> %h, <2 x half> %h)
  %add25 = add <2 x i16> %add23, %call24
  store <2 x i16> %add25, <2 x i16> addrspace(4)* %out, align 4
  ret void
}

declare spir_func <2 x i16> @_Z8isfiniteDv2_Dh(<2 x half>) local_unnamed_addr

declare spir_func <2 x i16> @_Z5isinfDv2_Dh(<2 x half>) local_unnamed_addr

declare spir_func <2 x i16> @_Z5isnanDv2_Dh(<2 x half>) local_unnamed_addr

declare spir_func <2 x i16> @_Z8isnormalDv2_Dh(<2 x half>) local_unnamed_addr

declare spir_func <2 x i16> @_Z7signbitDv2_Dh(<2 x half>) local_unnamed_addr

declare spir_func <2 x i16> @_Z7isequalDv2_DhS_(<2 x half>, <2 x half>) local_unnamed_addr

declare spir_func <2 x i16> @_Z10isnotequalDv2_DhS_(<2 x half>, <2 x half>) local_unnamed_addr

declare spir_func <2 x i16> @_Z9isgreaterDv2_DhS_(<2 x half>, <2 x half>) local_unnamed_addr

declare spir_func <2 x i16> @_Z14isgreaterequalDv2_DhS_(<2 x half>, <2 x half>) local_unnamed_addr

declare spir_func <2 x i16> @_Z6islessDv2_DhS_(<2 x half>, <2 x half>) local_unnamed_addr

declare spir_func <2 x i16> @_Z11islessequalDv2_DhS_(<2 x half>, <2 x half>) local_unnamed_addr

declare spir_func <2 x i16> @_Z13islessgreaterDv2_DhS_(<2 x half>, <2 x half>) local_unnamed_addr

declare spir_func <2 x i16> @_Z9isorderedDv2_DhS_(<2 x half>, <2 x half>) local_unnamed_addr

declare spir_func <2 x i16> @_Z11isunorderedDv2_DhS_(<2 x half>, <2 x half>) local_unnamed_addr
