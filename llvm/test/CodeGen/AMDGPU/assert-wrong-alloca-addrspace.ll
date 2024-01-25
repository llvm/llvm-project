; RUN: not --crash llc -mtriple=amdgcn -mcpu=gfx900 -filetype=null %s 2>&1 | FileCheck %s

; The alloca has the wrong address space and is passed to a call. The
; FrameIndex was created with the natural 32-bit pointer type instead
; of the declared 64-bit. Make sure we don't assert.

; CHECK: LLVM ERROR: Cannot select: {{.*}}: i64 = FrameIndex<0>

declare void @func(ptr)

define void @main() {
bb:
  %alloca = alloca i32, align 4
  call void @func(ptr %alloca)
  ret void
}
