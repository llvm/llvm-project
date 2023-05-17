; RUN: llc < %s | FileCheck -check-prefix=ENABLED %s
; RUN: llc -disable-nvptx-load-store-vectorizer < %s | FileCheck -check-prefix=DISABLED %s
; RUN: %if ptxas %{ llc < %s | %ptxas-verify %}
; RUN: %if ptxas %{ llc -disable-nvptx-load-store-vectorizer < %s | %ptxas-verify %}

target triple = "nvptx64-nvidia-cuda"

; Check that the load-store vectorizer is enabled by default for nvptx, and
; that it's disabled by the appropriate flag.

; ENABLED: ld.v2.{{.}}32
; DISABLED: ld.{{.}}32
; DISABLED: ld.{{.}}32
define i32 @f(ptr %p) {
  %p.1 = getelementptr i32, ptr %p, i32 1
  %v0 = load i32, ptr %p, align 8
  %v1 = load i32, ptr %p.1, align 4
  %sum = add i32 %v0, %v1
  ret i32 %sum
}

define half @fh(ptr %p) {
  %p.1 = getelementptr half, ptr %p, i32 1
  %p.2 = getelementptr half, ptr %p, i32 2
  %p.3 = getelementptr half, ptr %p, i32 3
  %p.4 = getelementptr half, ptr %p, i32 4
  %v0 = load half, ptr %p, align 64
  %v1 = load half, ptr %p.1, align 4
  %v2 = load half, ptr %p.2, align 4
  %v3 = load half, ptr %p.3, align 4
  %v4 = load half, ptr %p.4, align 4
  %sum1 = fadd half %v0, %v1
  %sum2 = fadd half %v2, %v3
  %sum3 = fadd half %sum1, %sum2
  %sum = fadd half %sum3, %v4
  ret half %sum
}

define float @ff(ptr %p) {
  %p.1 = getelementptr float, ptr %p, i32 1
  %p.2 = getelementptr float, ptr %p, i32 2
  %p.3 = getelementptr float, ptr %p, i32 3
  %p.4 = getelementptr float, ptr %p, i32 4
  %v0 = load float, ptr %p, align 64
  %v1 = load float, ptr %p.1, align 4
  %v2 = load float, ptr %p.2, align 4
  %v3 = load float, ptr %p.3, align 4
  %v4 = load float, ptr %p.4, align 4
  %sum1 = fadd float %v0, %v1
  %sum2 = fadd float %v2, %v3
  %sum3 = fadd float %sum1, %sum2
  %sum = fadd float %sum3, %v4
  ret float %sum
}
