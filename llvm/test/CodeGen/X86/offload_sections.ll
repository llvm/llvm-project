; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s --check-prefix=CHECK-ELF
; RUN: llc < %s -mtriple=x86_64-win32-gnu | FileCheck %s --check-prefix=CHECK-COFF

@llvm.embedded.object = private constant [1 x i8] c"\00", section ".llvm.offloading"
@llvm.compiler.used = appending global [1 x i8*] [i8* getelementptr inbounds ([1 x i8], [1 x i8]* @llvm.embedded.object, i32 0, i32 0)], section "llvm.metadata"

; CHECK-ELF: .section	.llvm.offloading,"e"
; CHECK-COFF: .section	.llvm.offloading,"dr"
