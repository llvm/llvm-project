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
