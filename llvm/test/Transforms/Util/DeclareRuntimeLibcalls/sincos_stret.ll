; RUN: %if x86-registered-target %{ opt -S -passes=declare-runtime-libcalls -mtriple=x86_64-apple-macos10.9 < %s | FileCheck -check-prefixes=CHECK,X64 %s %}
; RUN: %if aarch64-registered-target %{ opt -S -passes=declare-runtime-libcalls -mtriple=arm64-apple-macos10.9 < %s | FileCheck -check-prefixes=CHECK,STRUCT %s %}
; RUN: %if arm-registered-target %{ opt -S -passes=declare-runtime-libcalls -mtriple=thumbv7k-apple-watchos2.0 < %s | FileCheck -check-prefixes=CHECK,STRUCT %s %}
; RUN: %if arm-registered-target %{ opt -S -passes=declare-runtime-libcalls -mtriple=armv7-apple-ios7 < %s | FileCheck -check-prefix=SRET %s %}
; RUN: %if arm-registered-target %{ opt -S -passes=declare-runtime-libcalls -mtriple=thumbv7-apple-ios7 < %s | FileCheck -check-prefix=SRET %s %}

; RUN: %if arm-registered-target %{ opt -S -passes=declare-runtime-libcalls -mtriple=armv7-apple-ios6 < %s | FileCheck -check-prefix=NONE %s %}
; RUN: %if x86-registered-target %{ opt -S -passes=declare-runtime-libcalls -mtriple=x86_64-apple-macos10.8 < %s | FileCheck -check-prefix=NONE %s %}

; X64: declare { double, double } @__sincos_stret(double) [[SINCOS_ATTRS:#[0-9]+]]
; X64: declare <2 x float> @__sincosf_stret(float) [[SINCOS_ATTRS:#[0-9]+]]

; STRUCT: declare { double, double } @__sincos_stret(double) [[SINCOS_ATTRS:#[0-9]+]]
; STRUCT: declare { float, float } @__sincosf_stret(float) [[SINCOS_ATTRS:#[0-9]+]]

; SRET: declare void @__sincos_stret(ptr sret({ double, double }) align 4, double) [[SINCOS_ATTRS:#[0-9]+]]
; SRET: declare void @__sincosf_stret(ptr sret({ float, float }) align 4, float) [[SINCOS_ATTRS:#[0-9]+]]

; CHECK: attributes [[SINCOS_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(errnomem: write) }
; SRET: attributes [[SINCOS_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write, errnomem: write) }

; NONE-NOT: __sincos_stret
; NONE-NOT: __sincosf_stret
