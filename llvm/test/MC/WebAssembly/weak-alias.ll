; RUN: llc -mtriple wasm32-unknown-unknown-wasm -filetype=obj %s -o %t.o
; RUN: obj2yaml %t.o | FileCheck %s
; RUN: llvm-objdump -t %t.o | FileCheck --check-prefix=CHECK-SYMS %s

; 'foo_alias()' is weak alias of function 'foo()'
; 'bar_alias' is weak alias of global variable 'bar'
; Generates two exports of the same function, one of them weak

@bar = global i32 7, align 8
@bar_alias = weak hidden alias i32, i32* @bar
@bar_alias_address = global i32* @bar_alias, align 8

@foo_alias = weak hidden alias i32 (), i32 ()* @foo

define hidden i32 @call_alias() #0 {
entry:
  %call = call i32 @foo_alias()
  ret i32 %call
}

define hidden i32 @foo() #0 {
entry:
  ret i32 0
}


; CHECK:        - Type:            TYPE
; CHECK-NEXT:     Signatures:      
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         ReturnType:      I32
; CHECK-NEXT:         ParamTypes:      

; CHECK:        - Type:            IMPORT
; CHECK-NEXT:     Imports:         
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           foo_alias
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         SigIndex:        0
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           bar_alias
; CHECK-NEXT:         Kind:            GLOBAL
; CHECK-NEXT:         GlobalType:      I32
; CHECK-NEXT:         GlobalMutable:   false

; CHECK:        - Type:            FUNCTION
; CHECK-NEXT:     FunctionTypes:   [ 0, 0 ]

; CHECK:        - Type:            EXPORT
; CHECK-NEXT:     Exports:         
; CHECK-NEXT:       - Name:            call_alias
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:       - Name:            foo
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Index:           2
; CHECK-NEXT:       - Name:            bar
; CHECK-NEXT:         Kind:            GLOBAL
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:       - Name:            bar_alias_address
; CHECK-NEXT:         Kind:            GLOBAL
; CHECK-NEXT:         Index:           2
; CHECK-NEXT:       - Name:            foo_alias
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Index:           2
; CHECK-NEXT:       - Name:            bar_alias
; CHECK-NEXT:         Kind:            GLOBAL
; CHECK-NEXT:         Index:           1

; CHECK:        - Type:            DATA
; CHECK-NEXT:     Relocations:     
; CHECK-NEXT:       - Type:            R_WEBASSEMBLY_MEMORY_ADDR_I32
; CHECK-NEXT:         Index:           0
; CHECK-NEXT:         Offset:          0x0000000F
; CHECK-NEXT:     Segments:        
; CHECK-NEXT:       - SectionOffset:   6
; CHECK-NEXT:         MemoryIndex:     0
; CHECK-NEXT:         Offset:          
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           0
; CHECK-NEXT:         Content:         '07000000'
; CHECK-NEXT:       - SectionOffset:   15
; CHECK-NEXT:         MemoryIndex:     0
; CHECK-NEXT:         Offset:          
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           8
; CHECK-NEXT:         Content:         '00000000'

; CHECK:        - Type:            CUSTOM
; CHECK-NEXT:     Name:            name
; CHECK-NEXT:     FunctionNames:   
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Name:            foo_alias
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Name:            call_alias
; CHECK-NEXT:       - Index:           2
; CHECK-NEXT:         Name:            foo
; CHECK-NEXT:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            linking
; CHECK-NEXT:     DataSize:        12
; CHECK-NEXT:     DataAlignment:   8
; CHECK-NEXT:     SymbolInfo:      
; CHECK-NEXT:       - Name:            foo_alias
; CHECK-NEXT:         Flags:           1
; CHECK-NEXT:       - Name:            bar_alias
; CHECK-NEXT:         Flags:           1
; CHECK-NEXT:     SegmentNames:    
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Name:            .data.bar
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Name:            .data.bar_alias_address
; CHECK-NEXT: ...

; CHECK-SYMS: SYMBOL TABLE:
; CHECK-SYMS-NEXT: 00000000 g     F name	foo_alias
; CHECK-SYMS-NEXT: 00000001 g     F name	call_alias
; CHECK-SYMS-NEXT: 00000002 g     F name	foo
; CHECK-SYMS-NEXT: 00000002 gw    F EXPORT	foo_alias
; CHECK-SYMS-NEXT: 00000000 gw      EXPORT	bar_alias
; CHECK-SYMS-NEXT: 00000001 g     F EXPORT	call_alias
; CHECK-SYMS-NEXT: 00000002 g     F EXPORT	foo
; CHECK-SYMS-NEXT: 00000000 g       EXPORT	bar
; CHECK-SYMS-NEXT: 00000008 g       EXPORT	bar_alias_address
