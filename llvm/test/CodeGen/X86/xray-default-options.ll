; RUN: opt -S -p xray-preparation < %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

; CHECK: $__llvm_xray_options = comdat any
; CHECK: @__llvm_xray_options = hidden constant [40 x i8] c"patch_premain=true,xray_mode=xray-basic\00", comdat

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"xray-default-opts", !"patch_premain=true,xray_mode=xray-basic"}
