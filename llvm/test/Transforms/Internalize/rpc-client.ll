; The RPC client symbol must survive internalization on GPU targets, which use
; it for host callbacks, but is internalized like any other global elsewhere.

; RUN: opt -mtriple=amdgcn-amd-amdhsa < %s -passes=internalize -S \
; RUN:   | FileCheck %s --check-prefix=GPU
; RUN: opt -mtriple=nvptx64-nvidia-cuda < %s -passes=internalize -S \
; RUN:   | FileCheck %s --check-prefix=GPU
; RUN: opt -mtriple=x86_64-unknown-linux-gnu < %s -passes=internalize -S \
; RUN:   | FileCheck %s --check-prefix=HOST

; GPU: @__llvm_rpc_client = protected global i64 0, align 8
; HOST: @__llvm_rpc_client = internal global i64 0, align 8
@__llvm_rpc_client = protected global i64 zeroinitializer, align 8
