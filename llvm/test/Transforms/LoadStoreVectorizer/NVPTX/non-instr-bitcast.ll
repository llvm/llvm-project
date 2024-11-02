; RUN: opt -mtriple=nvptx64-nvidia-cuda -passes=load-store-vectorizer -S -o - %s | FileCheck %s

; Load from a constant.  This can be vectorized, but shouldn't crash us.

@global = internal addrspace(1) constant [4 x float] [float 0xBF71111120000000, float 0x3F70410420000000, float 0xBF81111120000000, float 0x3FB5555560000000], align 4

define void @foo() {
  ; CHECK: load <4 x float>
  %a = load float, float addrspace(1)* getelementptr inbounds ([4 x float], [4 x float] addrspace(1)* @global, i64 0, i64 0), align 16
  %b = load float, float addrspace(1)* getelementptr inbounds ([4 x float], [4 x float] addrspace(1)* @global, i64 0, i64 1), align 4
  %c = load float, float addrspace(1)* getelementptr inbounds ([4 x float], [4 x float] addrspace(1)* @global, i64 0, i64 2), align 4
  %d = load float, float addrspace(1)* getelementptr inbounds ([4 x float], [4 x float] addrspace(1)* @global, i64 0, i64 3), align 4
  ret void
}
