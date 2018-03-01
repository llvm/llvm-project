; RUN: llc -O2 -filetype=obj %s -o - | llvm-readobj -r -s -expand-relocs | FileCheck %s

target triple = "wasm32-unknown-unknown-wasm"

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
  %adr = alloca i32 ()*, align 4
  store i32 ()* @import3, i32 ()** %adr, align 4
  ret void
}

; CHECK:   Section {
; CHECK:     Type: ELEM (0x9)
; CHECK:     Size: 7
; CHECK:   }

; CHECK: Relocations [
; CHECK:   Section (5) CODE {
; CHECK:     Relocation {
; CHECK:       Type: R_WEBASSEMBLY_FUNCTION_INDEX_LEB (0)
; CHECK:       Offset: 0x4
; CHECK:       Index: 0x1
; CHECK:     }
; CHECK:     Relocation {
; CHECK:       Type: R_WEBASSEMBLY_FUNCTION_INDEX_LEB (0)
; CHECK:       Offset: 0xB
; CHECK:       Index: 0x2
; CHECK:     }
; CHECK:     Relocation {
; CHECK:       Type: R_WEBASSEMBLY_TABLE_INDEX_SLEB (1)
; CHECK:       Offset: 0x1E
; CHECK:       Index: 0x5
; CHECK:     }
; CHECK:   }
