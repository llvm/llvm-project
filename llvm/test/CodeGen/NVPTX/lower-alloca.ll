; RUN: opt < %s -S -nvptx-lower-alloca -infer-address-spaces | FileCheck %s
; RUN: opt < %s -S -nvptx-lower-alloca | FileCheck %s --check-prefix LOWERALLOCAONLY
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_35 | FileCheck %s --check-prefix PTX
; RUN: llc < %s -O0 -mtriple=nvptx64 -mcpu=sm_35 | FileCheck %s --check-prefix PTXO0
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_35 | %ptxas-verify %}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-unknown-unknown"

define ptx_kernel void @kernel() {
; LABEL: @lower_alloca
; PTX-LABEL: .visible .entry kernel(
; PTXO0-LABEL: .visible .entry kernel(
  %A = alloca i32
; CHECK: [[A:%.*]] = alloca i32, align 4, addrspace(5)
; CHECK: [[GENERIC:%.*]] = addrspacecast ptr addrspace(5) [[A]] to ptr
; CHECK: store i32 0, ptr addrspace(5) [[A]]
; CHECK: call void @callee(ptr [[GENERIC]])
; LOWERALLOCAONLY: [[A:%.*]] = alloca i32, align 4, addrspace(5)
; LOWERALLOCAONLY: [[GENERIC:%.*]] = addrspacecast ptr addrspace(5) [[A]] to ptr
; LOWERALLOCAONLY: store i32 0, ptr [[GENERIC]], align 4
; LOWERALLOCAONLY: call void @callee(ptr [[GENERIC]])
; PTX: st.local.b32 [%SPL], 0
; PTXO0: mov.b64 %SPL, __local_depot0;
; PTXO0: cvta.local.u64 [[SP:%rd[0-9]+]], {{%rd[0-9]+}};
; PTXO0: st.b32 {{\[}}[[SP]]{{\]}}, 0;
  store i32 0, ptr %A
  call void @callee(ptr %A)
  ret void
}

define void @alloca_in_explicit_local_as() {
; LABEL: @lower_alloca_addrspace5
; PTX-LABEL: .visible .func alloca_in_explicit_local_as(
  %A = alloca i32, addrspace(5)
; CHECK: store i32 0, ptr addrspace(5) {{%.+}}
; PTX: st.local.b32 [%SPL], 0
; An alloca already in the local address space is left as-is.
; LOWERALLOCAONLY: store i32 0, ptr addrspace(5) %A, align 4
  store i32 0, ptr addrspace(5) %A
  call void @callee(ptr addrspace(5) %A)
  ret void
}

; Lifetime markers must keep referencing the alloca itself (the verifier rejects
; a cast operand), so they are retargeted to the local alloca and their overload
; is updated to the local address space, rather than going through the cast.
define void @lifetime_alloca() {
; PTX-LABEL: .visible .func lifetime_alloca(
  %A = alloca i32
; CHECK: [[A:%.*]] = alloca i32, align 4, addrspace(5)
; CHECK: call void @llvm.lifetime.start.p5(ptr addrspace(5) [[A]])
; CHECK: store i32 0, ptr addrspace(5) [[A]]
; CHECK: call void @llvm.lifetime.end.p5(ptr addrspace(5) [[A]])
; LOWERALLOCAONLY: [[A:%.*]] = alloca i32, align 4, addrspace(5)
; LOWERALLOCAONLY: call void @llvm.lifetime.start.p5(ptr addrspace(5) [[A]])
; LOWERALLOCAONLY: call void @llvm.lifetime.end.p5(ptr addrspace(5) [[A]])
; PTX: st.local.b32 [%SPL], 0
  call void @llvm.lifetime.start.p0(ptr %A)
  store i32 0, ptr %A
  call void @callee(ptr %A)
  call void @llvm.lifetime.end.p0(ptr %A)
  ret void
}

; Debug records are retargeted to the local alloca too, so the variable stays
; described by its stack slot rather than the cvta.local result.
define void @dbg_alloca() !dbg !10 {
  %A = alloca i32
; CHECK: [[A:%.*]] = alloca i32, align 4, addrspace(5)
; CHECK: #dbg_declare(ptr addrspace(5) [[A]],
; LOWERALLOCAONLY: [[A:%.*]] = alloca i32, align 4, addrspace(5)
; LOWERALLOCAONLY: #dbg_declare(ptr addrspace(5) [[A]],
  call void @llvm.dbg.declare(metadata ptr %A, metadata !13, metadata !DIExpression()), !dbg !15
  store i32 0, ptr %A
  ret void
}

declare void @callee(ptr)
declare void @callee_addrspace5(ptr addrspace(5))
declare void @llvm.lifetime.start.p0(ptr)
declare void @llvm.lifetime.end.p0(ptr)
declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!5}
!nvvm.annotations = !{!1}
!1 = !{ptr @alloca_in_explicit_local_as, !"alloca_in_explicit_local_as", i32 1}
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, emissionKind: FullDebug)
!3 = !DIFile(filename: "lower-alloca.c", directory: "/")
!5 = !{i32 2, !"Debug Info Version", i32 3}
!10 = distinct !DISubprogram(name: "dbg_alloca", scope: !3, file: !3, line: 1, type: !11, unit: !2)
!11 = !DISubroutineType(types: !12)
!12 = !{null}
!13 = !DILocalVariable(name: "x", scope: !10, file: !3, line: 1, type: !14)
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DILocation(line: 1, column: 1, scope: !10)
