; RUN: llvm-ml -m32 %s /Fo - | llvm-readobj --syms - | FileCheck %s --check-prefix=CHECK-OBJ --check-prefix=CHECK-OBJ-NOSAFESEH
; RUN: llvm-ml -m64 %s /Fo - | llvm-readobj --syms - | FileCheck %s --check-prefix=CHECK-OBJ --check-prefix=CHECK-OBJ-NOSAFESEH

; RUN: llvm-ml -m32 -safeseh %s /Fo - | llvm-readobj --syms - | FileCheck %s --check-prefix=CHECK-OBJ --check-prefix=CHECK-OBJ-SAFESEH
; RUN: llvm-ml -m64 -safeseh %s /Fo %t.obj 2>&1 | FileCheck %s --check-prefix=CHECK-SAFESEH64
; RUN: llvm-readobj --syms %t.obj | FileCheck %s --check-prefix=CHECK-OBJ --check-prefix=CHECK-OBJ-NOSAFESEH

; CHECK-SAFESEH64: warning: /safeseh applies only to 32-bit X86 platforms; ignoring.

.code
noop:
  ret
end

; CHECK-OBJ:       Symbol {
; CHECK-OBJ:         Name: @feat.00
; CHECK-OBJ-NOSAFESEH:  Value: 2
; CHECK-OBJ-SAFESEH:    Value: 3
; CHECK-OBJ-NEXT:    Section: IMAGE_SYM_ABSOLUTE
; CHECK-OBJ-NEXT:    BaseType: Null
; CHECK-OBJ-NEXT:    ComplexType: Null
; CHECK-OBJ-NEXT:    StorageClass: External
; CHECK-OBJ-NEXT:    AuxSymbolCount: 0
; CHECK-OBJ-NEXT:  }
