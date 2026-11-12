; RUN: llc < %s -asm-verbose=false -wasm-keep-registers | FileCheck %s
; RUN: llc < %s --filetype=obj | obj2yaml | FileCheck --check-prefix=OBJ %s

target triple = "wasm32-unknown-unknown"

@imported_g = external addrspace(1) global i32 "wasm-import-module"="js" "wasm-import-name"="global_g"

define void @test() {
  call void @foo()
  call void @plain()
  ret void
}

define i32 @get() {
  %v = load i32, ptr addrspace(1) @imported_g
  ret i32 %v
}

declare void @foo() "wasm-import-module"="bar" "wasm-import-name"="qux"
declare void @plain()

; CHECK-NOT: .import_module plain
;     CHECK: .import_module foo, "bar"
;     CHECK: .import_name foo, "qux"
; CHECK-NOT: .import_module plain

; CHECK: .globaltype imported_g, i32
; CHECK-NEXT: .import_module imported_g, "js"
; CHECK-NEXT: .import_name imported_g, "global_g"

; OBJ:      - Type:            IMPORT
; OBJ:        Imports:
; OBJ:          - Module:          bar
; OBJ-NEXT:       Field:           qux
; OBJ-NEXT:       Kind:            FUNCTION
; OBJ:          - Module:          js
; OBJ-NEXT:       Field:           global_g
; OBJ-NEXT:       Kind:            GLOBAL
; OBJ-NEXT:       GlobalType:      I32
; OBJ-NEXT:       GlobalMutable:   true
