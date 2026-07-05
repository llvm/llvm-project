; RUN: llc -O2 -filetype=obj %s -o - | llvm-readobj -S -r --expand-relocs - | FileCheck %s

target triple = "wasm32-unknown-unknown"

declare i32 @import1()
declare i32 @import2()
declare i32 @import3()

; call the imports to make sure they are included in the imports section
define hidden void @call_imports() #0 {
entry:
  %call = call i32 @import1()
  %call1 = call i32 @import2()
  ret void
}

; take the address of the third import.  This should generate a TABLE_INDEX
; relocation with index of 0 since its the first and only address taken
; function.
define hidden void @call_indirect() #0 {
entry:
  %adr = alloca ptr, align 4
  store ptr @import3, ptr %adr, align 4
  ret void
}

; CHECK:          Type: ELEM (0x9)
; CHECK-NEXT:     Size: 7

; CHECK:      Relocations [
; CHECK-NEXT:   Section (5) CODE {
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Type: R_WASM_FUNCTION_INDEX_LEB (0)
; CHECK-NEXT:       Offset: 0x4
; CHECK-NEXT:       Symbol: import1
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Type: R_WASM_FUNCTION_INDEX_LEB (0)
; CHECK-NEXT:       Offset: 0xB
; CHECK-NEXT:       Symbol: import2
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Type: R_WASM_GLOBAL_INDEX_LEB (7)
; CHECK-NEXT:       Offset: 0x15
; CHECK-NEXT:       Symbol: __stack_pointer
; CHECK-NEXT:     }
; CHECK-NEXT:     Relocation {
; CHECK-NEXT:       Type: R_WASM_TABLE_INDEX_SLEB (1)
; CHECK-NEXT:       Offset: 0x1E
; CHECK-NEXT:       Symbol: import3
; CHECK-NEXT:     }
; CHECK-NEXT:   }
; CHECK-NEXT: ]
