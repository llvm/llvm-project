; RUN: llc < %s | FileCheck %s
target triple = "i686-pc-win32"

declare i32 @llvm.x86.flags.read.u32()
declare void @llvm.x86.flags.write.u32(i32)

define i32 @read_flags() {
entry:
  %flags = call i32 @llvm.x86.flags.read.u32()
  ret i32 %flags
}

; CHECK-LABEL: _read_flags:
; CHECK:      pushl   %ebp
; CHECK-NEXT: movl    %esp, %ebp
; CHECK-NEXT: pushfl
; CHECK-NEXT: popl    %eax
; CHECK-NEXT: popl    %ebp

define x86_fastcallcc void @write_flags(i32 inreg %arg) {
entry:
  call void @llvm.x86.flags.write.u32(i32 %arg)
  ret void
}

; CHECK-LABEL: @write_flags@4:
; CHECK:      pushl   %ebp
; CHECK-NEXT: movl    %esp, %ebp
; CHECK-NEXT: pushl   %ecx
; CHECK-NEXT: popfl
; CHECK-NEXT: popl    %ebp
