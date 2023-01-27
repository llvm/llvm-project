; RUN: opt < %s -S -nvptx-lower-args --mtriple nvptx64-nvidia-cuda | FileCheck %s --check-prefixes COMMON,IR,IRC
; RUN: opt < %s -S -nvptx-lower-args --mtriple nvptx64-nvidia-nvcl | FileCheck %s --check-prefixes COMMON,IR,IRO
; RUN: llc < %s -mcpu=sm_20 --mtriple nvptx64-nvidia-cuda | FileCheck %s --check-prefixes COMMON,PTX,PTXC
; RUN: llc < %s -mcpu=sm_20 --mtriple nvptx64-nvidia-nvcl| FileCheck %s --check-prefixes COMMON,PTX,PTXO
; RUN: %if ptxas %{ llc < %s -mcpu=sm_20 | %ptxas-verify %}

target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%class.outer = type <{ %class.inner, i32, [4 x i8] }>
%class.inner = type { ptr, ptr }

; Check that nvptx-lower-args preserves arg alignment
; COMMON-LABEL: load_alignment
define void @load_alignment(ptr nocapture readonly byval(%class.outer) align 8 %arg) {
entry:
; IR: load %class.outer, ptr addrspace(101)
; IR-SAME: align 8
; PTX: ld.param.u64
; PTX-NOT: ld.param.u8
  %arg.idx.val = load ptr, ptr %arg, align 8
  %arg.idx1 = getelementptr %class.outer, ptr %arg, i64 0, i32 0, i32 1
  %arg.idx1.val = load ptr, ptr %arg.idx1, align 8
  %arg.idx2 = getelementptr %class.outer, ptr %arg, i64 0, i32 1
  %arg.idx2.val = load i32, ptr %arg.idx2, align 8
  %arg.idx.val.val = load i32, ptr %arg.idx.val, align 4
  %add.i = add nsw i32 %arg.idx.val.val, %arg.idx2.val
  store i32 %add.i, ptr %arg.idx1.val, align 4

  ; let the pointer escape so we still create a local copy this test uses to
  ; check the load alignment.
  %tmp = call ptr @escape(ptr nonnull %arg.idx2)
  ret void
}


; COMMON-LABEL: ptr_generic
define void @ptr_generic(ptr %out, ptr %in) {
; IRC:  %in3 = addrspacecast ptr %in to ptr addrspace(1)
; IRC:  %in4 = addrspacecast ptr addrspace(1) %in3 to ptr
; IRC:  %out1 = addrspacecast ptr %out to ptr addrspace(1)
; IRC:  %out2 = addrspacecast ptr addrspace(1) %out1 to ptr
; PTXC: cvta.to.global.u64
; PTXC: cvta.to.global.u64
; PTXC: ld.global.u32
; PTXC: st.global.u32

; OpenCL can't make assumptions about incoming pointer, so we should generate
; generic pointers load/store.
; IRO-NOT: addrspacecast
; PTXO-NOT: cvta.to.global
; PTXO: ld.u32
; PTXO: st.u32
  %v = load i32, ptr  %in, align 4
  store i32 %v, ptr %out, align 4
  ret void
}

; COMMON-LABEL: ptr_nongeneric
define void @ptr_nongeneric(ptr addrspace(1) %out, ptr addrspace(4) %in) {
; IR-NOT: addrspacecast
; PTX-NOT: cvta.to.global
; PTX:  ld.const.u32
; PTX   st.global.u32
  %v = load i32, ptr addrspace(4) %in, align 4
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}


; Function Attrs: convergent nounwind
declare dso_local ptr @escape(ptr) local_unnamed_addr
!nvvm.annotations = !{!0, !1}
!0 = !{ptr @ptr_generic, !"kernel", i32 1}
!1 = !{ptr @ptr_nongeneric, !"kernel", i32 1}
