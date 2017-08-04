; RUN: llc < %s | FileCheck -check-prefix=ENABLED %s
; RUN: llc -disable-nvptx-load-store-vectorizer < %s | FileCheck -check-prefix=DISABLED %s
target triple = "nvptx64-nvidia-cuda"

; Check that the load-store vectorizer is enabled by default for nvptx, and
; that it's disabled by the appropriate flag.

; ENABLED: ld.v2.{{.}}32
; DISABLED: ld.{{.}}32
; DISABLED: ld.{{.}}32
define i32 @f(i32* %p) {
  %p.1 = getelementptr i32, i32* %p, i32 1
  %v0 = load i32, i32* %p, align 8
  %v1 = load i32, i32* %p.1, align 4
  %sum = add i32 %v0, %v1
  ret i32 %sum
}

define half @fh(half* %p) {
  %p.1 = getelementptr half, half* %p, i32 1
  %p.2 = getelementptr half, half* %p, i32 2
  %p.3 = getelementptr half, half* %p, i32 3
  %p.4 = getelementptr half, half* %p, i32 4
  %v0 = load half, half* %p, align 64
  %v1 = load half, half* %p.1, align 4
  %v2 = load half, half* %p.2, align 4
  %v3 = load half, half* %p.3, align 4
  %v4 = load half, half* %p.4, align 4
  %sum1 = fadd half %v0, %v1
  %sum2 = fadd half %v2, %v3
  %sum3 = fadd half %sum1, %sum2
  %sum = fadd half %sum3, %v4
  ret half %sum
}

define float @ff(float* %p) {
  %p.1 = getelementptr float, float* %p, i32 1
  %p.2 = getelementptr float, float* %p, i32 2
  %p.3 = getelementptr float, float* %p, i32 3
  %p.4 = getelementptr float, float* %p, i32 4
  %v0 = load float, float* %p, align 64
  %v1 = load float, float* %p.1, align 4
  %v2 = load float, float* %p.2, align 4
  %v3 = load float, float* %p.3, align 4
  %v4 = load float, float* %p.4, align 4
  %sum1 = fadd float %v0, %v1
  %sum2 = fadd float %v2, %v3
  %sum3 = fadd float %sum1, %sum2
  %sum = fadd float %sum3, %v4
  ret float %sum
}
