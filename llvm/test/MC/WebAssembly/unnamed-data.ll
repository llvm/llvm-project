; RUN: llc -mtriple wasm32-unknown-unknown-wasm -filetype=obj %s -o - | obj2yaml | FileCheck %s

@.str1 = private unnamed_addr constant [6 x i8] c"hello\00", align 1
@.str2 = private unnamed_addr constant [6 x i8] c"world\00", align 1

@a = global i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str1, i32 0, i32 0), align 8
@b = global i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str2, i32 0, i32 0), align 8


; CHECK:   - Type:            GLOBAL
; CHECK:     Globals:         
; CHECK:       - Type:            I32
; CHECK:         Mutable:         false
; CHECK:         InitExpr:        
; CHECK:           Opcode:          I32_CONST
; CHECK:           Value:           0
; CHECK:       - Type:            I32
; CHECK:         Mutable:         false
; CHECK:         InitExpr:        
; CHECK:           Opcode:          I32_CONST
; CHECK:           Value:           6
; CHECK:       - Type:            I32
; CHECK:         Mutable:         false
; CHECK:         InitExpr:        
; CHECK:           Opcode:          I32_CONST
; CHECK:           Value:           16
; CHECK:       - Type:            I32
; CHECK:         Mutable:         false
; CHECK:         InitExpr:        
; CHECK:           Opcode:          I32_CONST
; CHECK:           Value:           24
; CHECK:   - Type:            EXPORT
; CHECK:     Exports:         
; CHECK:       - Name:            a
; CHECK:         Kind:            GLOBAL
; CHECK:         Index:           2
; CHECK:       - Name:            b
; CHECK:         Kind:            GLOBAL
; CHECK:         Index:           3
; CHECK:   - Type:            DATA
; CHECK:     Relocations:     
; CHECK:       - Type:            R_WEBASSEMBLY_GLOBAL_ADDR_I32
; CHECK:         Index:           0
; CHECK:         Offset:          0x00000016
; CHECK:       - Type:            R_WEBASSEMBLY_GLOBAL_ADDR_I32
; CHECK:         Index:           1
; CHECK:         Offset:          0x0000001E
; CHECK:     Segments:        
; CHECK:       - Index:           0
; CHECK:         Offset:          
; CHECK:           Opcode:          I32_CONST
; CHECK:           Value:           0
; CHECK:         Content:         68656C6C6F00776F726C640000000000000000000000000006000000
