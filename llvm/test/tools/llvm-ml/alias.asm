; RUN: llvm-ml %s /Fo - | llvm-readobj --syms - | FileCheck %s

.code

proc1 PROC
  ret
proc1 ENDP

proc2 PROC
  ret
proc2 ENDP

alias <t1> = <proc1>
; CHECK:      Symbol {
; CHECK:        Name: t1
; CHECK-NEXT:   Value: 0
; CHECK-NEXT:   Section: IMAGE_SYM_UNDEFINED (0)
; CHECK-NEXT:   BaseType: Null
; CHECK-NEXT:   ComplexType: Null
; CHECK-NEXT:   StorageClass: WeakExternal
; CHECK-NEXT:   AuxSymbolCount: 1
; CHECK-NEXT:   AuxWeakExternal {
; CHECK-NEXT:     Linked: proc1
; CHECK-NEXT:     Search: Alias
; CHECK-NEXT:   }
; CHECK-NEXT: }

alias <t2> = <proc2>
; CHECK:      Symbol {
; CHECK:        Name: t2
; CHECK-NEXT:   Value: 0
; CHECK-NEXT:   Section: IMAGE_SYM_UNDEFINED (0)
; CHECK-NEXT:   BaseType: Null
; CHECK-NEXT:   ComplexType: Null
; CHECK-NEXT:   StorageClass: WeakExternal
; CHECK-NEXT:   AuxSymbolCount: 1
; CHECK-NEXT:   AuxWeakExternal {
; CHECK-NEXT:     Linked: proc2
; CHECK-NEXT:     Search: Alias
; CHECK-NEXT:   }
; CHECK-NEXT: }

alias <t3> = <foo>
; CHECK:      Symbol {
; CHECK:        Name: t3
; CHECK-NEXT:   Value: 0
; CHECK-NEXT:   Section: IMAGE_SYM_UNDEFINED (0)
; CHECK-NEXT:   BaseType: Null
; CHECK-NEXT:   ComplexType: Null
; CHECK-NEXT:   StorageClass: WeakExternal
; CHECK-NEXT:   AuxSymbolCount: 1
; CHECK-NEXT:   AuxWeakExternal {
; CHECK-NEXT:     Linked: foo
; CHECK-NEXT:     Search: Alias
; CHECK-NEXT:   }
; CHECK-NEXT: }

alias <t4> = <bar>
bar PROC
  ret
bar ENDP

; CHECK:      Symbol {
; CHECK:        Name: t4
; CHECK-NEXT:   Value: 0
; CHECK-NEXT:   Section: IMAGE_SYM_UNDEFINED (0)
; CHECK-NEXT:   BaseType: Null
; CHECK-NEXT:   ComplexType: Null
; CHECK-NEXT:   StorageClass: WeakExternal
; CHECK-NEXT:   AuxSymbolCount: 1
; CHECK-NEXT:   AuxWeakExternal {
; CHECK-NEXT:     Linked: bar
; CHECK-NEXT:     Search: Alias
; CHECK-NEXT:   }
; CHECK-NEXT: }

alias <t5> = <t2>
; CHECK:      Symbol {
; CHECK:        Name: t5
; CHECK-NEXT:   Value: 0
; CHECK-NEXT:   Section: IMAGE_SYM_UNDEFINED (0)
; CHECK-NEXT:   BaseType: Null
; CHECK-NEXT:   ComplexType: Null
; CHECK-NEXT:   StorageClass: WeakExternal
; CHECK-NEXT:   AuxSymbolCount: 1
; CHECK-NEXT:   AuxWeakExternal {
; CHECK-NEXT:     Linked: t2
; CHECK-NEXT:     Search: Alias
; CHECK-NEXT:   }
; CHECK-NEXT: }

END
