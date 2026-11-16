; RUN: opt < %s -passes=pgo-icall-prom -icp-samplepgo -icp-allow-decls -S | FileCheck %s

; Test that when indirect call promotion introduces an address comparison
; against a symbol (e.g. WrongSymbol, attributed by AutoFDO due to ICF folding),
; unnamed_addr is stripped from the callee symbol so that it is address-significant.

; CHECK: declare void @WrongSymbol(){{$}}
declare void @WrongSymbol() unnamed_addr

; CHECK: declare void @RightSymbol() unnamed_addr
declare void @RightSymbol() unnamed_addr

; CHECK-LABEL: define void @caller(
; CHECK:         %[[CMP:.*]] = icmp eq ptr %fp, @WrongSymbol
; CHECK-NEXT:    br i1 %[[CMP]], label %[[THEN:.*]], label %[[ELSE:.*]]
; CHECK:       [[THEN]]:
; CHECK-NEXT:    call void @WrongSymbol()
define void @caller(ptr %fp) {
  call void %fp(), !prof !0
  ret void
}

!0 = !{!"VP", i32 0, i64 1000, i64 6010974106627529467, i64 1000}
