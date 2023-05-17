; Test that symbols starting with '?' are not affected by IR mangling.

; RUN: llc -mtriple i686-pc-win32 < %s | FileCheck %s --check-prefix=COFF
; RUN: llc -mtriple x86_64-pc-win32 < %s | FileCheck %s --check-prefix=COFF64
; RUN: llc -mtriple i686-linux-gnu < %s | FileCheck %s --check-prefix=ELF
; RUN: llc -mtriple i686-apple-darwin < %s | FileCheck %s --check-prefix=MACHO

; Currently all object files allow escaping private symbols, but eventually we
; might want to reject that.

; COFF: calll "?withescape@A@@QBEXXZ"
; COFF: calll "?withquestion@A@@QBEXXZ"
; COFF: calll "L?privatequestion@A@@QBEXXZ"
; COFF: calll "L?privatequestionfast@A@@QBEXXZ"
; COFF: calll "?escapedprivate@A@@QBEXXZ"

; COFF64: callq "?withescape@A@@QBEXXZ"
; COFF64: callq "?withquestion@A@@QBEXXZ"
; COFF64: callq ".L?privatequestion@A@@QBEXXZ"
; COFF64: callq ".L?privatequestionfast@A@@QBEXXZ"
; COFF64: callq "?escapedprivate@A@@QBEXXZ"

; ELF: calll "?withescape@A@@QBEXXZ"
; ELF: calll "?withquestion@A@@QBEXXZ"
; ELF: calll ".L?privatequestion@A@@QBEXXZ"
; ELF: calll ".L?privatequestionfast@A@@QBEXXZ"
; ELF: calll "?escapedprivate@A@@QBEXXZ"

; MACHO: calll "?withescape@A@@QBEXXZ"
; MACHO: calll "_?withquestion@A@@QBEXXZ"
; MACHO: calll "l_?privatequestion@A@@QBEXXZ"
; MACHO: calll "l_?privatequestionfast@A@@QBEXXZ"
; MACHO: calll "?escapedprivate@A@@QBEXXZ"

%struct.A = type {}

define i32 @main() {
entry:
  tail call void @"\01?withescape@A@@QBEXXZ"(ptr null)
  tail call void @"?withquestion@A@@QBEXXZ"(ptr null)
  tail call void @"?privatequestion@A@@QBEXXZ"(ptr null)
  tail call x86_fastcallcc void @"?privatequestionfast@A@@QBEXXZ"(ptr null)
  tail call void @"\01?escapedprivate@A@@QBEXXZ"(ptr null)
  ret i32 0
}

declare void @"\01?withescape@A@@QBEXXZ"(ptr)
declare void @"?withquestion@A@@QBEXXZ"(ptr)

define private void @"?privatequestion@A@@QBEXXZ"(ptr) {
  ret void
}

define private x86_fastcallcc void @"?privatequestionfast@A@@QBEXXZ"(ptr) {
  ret void
}

define private void @"\01?escapedprivate@A@@QBEXXZ"(ptr) {
  ret void
}
