; RUN: opt --mtriple=x86_64-unknown-linux-gnu -passes='embed-bitcode' -S < %s | FileCheck %s
; RUN: opt --mtriple=x86_64-unknown-linux-gnu -passes='fatlto-pre-link<O2;thinlto>' -S < %s | FileCheck %s

@llvm.embedded.module = private constant [4 x i8] c"BC\C0\DE", section ".llvmbc", align 1
@llvm.cmdline = private constant [5 x i8] c"-cc1\00", section ".llvmcmd", align 1
@llvm.compiler.used = appending global [2 x ptr] [ptr @llvm.embedded.module, ptr @llvm.cmdline], section "llvm.metadata"

define i32 @foo() {
  ret i32 7
}

; CHECK: @llvm.embedded.object = private constant {{.*}} section ".llvm.lto"
; CHECK: @llvm.embedded.module = private constant [4 x i8] c"BC\C0\DE", section ".llvmbc", align 1
; CHECK: @llvm.cmdline = private constant [5 x i8] c"-cc1\00", section ".llvmcmd", align 1
; CHECK: @llvm.compiler.used = appending global [3 x ptr] [ptr @llvm.embedded.object, ptr @llvm.embedded.module, ptr @llvm.cmdline], section "llvm.metadata"
