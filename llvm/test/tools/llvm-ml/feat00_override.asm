; RUN: llvm-ml %s /Fo - | llvm-readobj --syms - | FileCheck %s
; RUN: llvm-ml64 %s /Fo - | llvm-readobj --syms - | FileCheck %s
; RUN: llvm-ml -safeseh %s /Fo - | llvm-readobj --syms - | FileCheck %s

.code

@feat.00 = 99

noop:
  ret
end

; CHECK:       Symbol {
; CHECK:         Name: @feat.00
; CHECK:         Value: 99
; CHECK-NEXT:    Section: IMAGE_SYM_ABSOLUTE
; CHECK-NEXT:    BaseType: Null
; CHECK-NEXT:    ComplexType: Null
; CHECK-NEXT:    StorageClass: Static
; CHECK-NEXT:    AuxSymbolCount: 0
; CHECK-NEXT:  }
