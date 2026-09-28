; RUN: llc -mtriple s390x-zos -filetype=obj %s -o %t.o
; RUN: llc -mtriple s390x-zos < %s | FileCheck %s

define { ptr, i32 } @fn_with_lpad() personality ptr @__zos_cxx_personality_v2 {
start:
  %_0 = invoke double null(double 0.0, double 0.0, double 0.0)
          to label %bb1 unwind label %terminate
terminate:
  %0 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  ret { ptr, i32 } %0
bb1:
  ret { ptr, i32 } zeroinitializer
}

define double @fn_with_const_pool() {
start:
  ret double 1.0   ; forces a constant pool entry (L#CPI1_0)
}

declare i32 @__zos_cxx_personality_v2(...)

; CHECK:      C_WSA64 CATTR ALIGN(2),FILL(0),NOTEXECUTABLE,RMODE(64),PART(.gcc_excepti
; CHECK-NEXT:                ion_table.fn_with_lpad)
; CHECK-NEXT: .gcc_exception_table.fn_with_lpad XATTR LINKAGE(XPLINK),REFERENCE(DATA),
; CHECK-NEXT:                ,SCOPE(SECTION)
; CHECK-NEXT:  DS 0B
; CHECK-NEXT: * @LPStart Encoding = omit
; CHECK-NEXT:  DC XL1'FF'
; CHECK-NEXT: * @TType Encoding = absptr
; CHECK-NEXT:  DC XL1'00'
; CHECK-NEXT: * Call site Encoding = uleb128
; CHECK-NEXT:  DC XL1'01'
; CHECK-NEXT: * >> Call Site 1 <<
; CHECK-NEXT: *   Call between L#tmp0 and L#tmp1
; CHECK-NEXT: *     jumps to L#tmp2
; CHECK-NEXT: *   On action: 1
; CHECK-NEXT:  DC XL1'01'
; CHECK-NEXT: * >> Action Record 1 <<
; CHECK-NEXT: *   Filter TypeInfo -1
; CHECK-NEXT:  DC XL1'7F'
; CHECK-NEXT: *   No further actions
; CHECK-NEXT:  DC XL1'00'
; CHECK-NEXT:  DS 0B
; CHECK-NEXT: * >> Filter TypeInfos <<
; CHECK-NEXT:  DC XL1'00'
; CHECK-NEXT:  DS 0B
