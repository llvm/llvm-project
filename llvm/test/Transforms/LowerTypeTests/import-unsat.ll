; Test that we correctly import an unsat resolution for type identifier "typeid1".
; RUN: opt -S -passes=lowertypetests -lowertypetests-summary-action=import -lowertypetests-read-summary=%S/Inputs/import-unsat.yaml -lowertypetests-write-summary=%t %s | FileCheck %s
; RUN: FileCheck --check-prefix=SUMMARY %s < %t

; SUMMARY:      GlobalValueMap:
; SUMMARY-NEXT:   42:
; SUMMARY-NEXT:    - Linkage:             0
; SUMMARY-NEXT:      Visibility:          0
; SUMMARY-NEXT:      NotEligibleToImport: false
; SUMMARY-NEXT:      Live:                true
; SUMMARY-NEXT:      Local:               false
; SUMMARY-NEXT:      CanAutoHide:         false
; SUMMARY-NEXT:      TypeTests: [ 123 ]
; SUMMARY-NEXT: TypeIdMap:
; SUMMARY-NEXT:   typeid1:
; SUMMARY-NEXT:     TTRes:
; SUMMARY-NEXT:       Kind:            Unsat
; SUMMARY-NEXT:       SizeM1BitWidth:  0

target datalayout = "e-p:32:32"

declare i1 @llvm.type.test(i8* %ptr, metadata %bitset) nounwind readnone

define i1 @foo(i8* %p) {
  %x = call i1 @llvm.type.test(i8* %p, metadata !"typeid1")
  ; CHECK: ret i1 false
  ret i1 %x
}
