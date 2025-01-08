; RUN: llc -mtriple mipsel-windows -filetype=obj < %s | obj2yaml | FileCheck %s

; CHECK:  Machine:         IMAGE_FILE_MACHINE_R4000



; CHECK:  - Name:            .text
; CHECK:    Relocations:

declare void @bar()
define i32 @foo_jmp() {
  call i32 @bar()
; CHECK:      - VirtualAddress:  8
; CHECK:        SymbolName:      bar
; CHECK:        Type:            IMAGE_REL_MIPS_JMPADDR
  ret i32 0
}

@var = external global i32
define i32 @foo_var() {
  %1 = load i32, i32* @var
; CHECK:      - VirtualAddress:  32
; CHECK:        SymbolName:      var
; CHECK:        Type:            IMAGE_REL_MIPS_REFHI
; CHECK:      - VirtualAddress:  40
; CHECK:        SymbolName:      var
; CHECK:        Type:            IMAGE_REL_MIPS_REFLO
  ret i32 %1
}



; CHECK:  - Name:            .data
; CHECK:    Relocations:

%struct._PTR = type { ptr }

@var1 = internal global %struct._PTR { ptr @var2 }
@var2 = external global i32
; CHECK:      - VirtualAddress:  0
; CHECK:        SymbolName:      var2
; CHECK:        Type:            IMAGE_REL_MIPS_REFWORD
