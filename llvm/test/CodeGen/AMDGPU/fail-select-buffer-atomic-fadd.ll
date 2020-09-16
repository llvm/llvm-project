; RUN: not --crash llc -march=amdgcn -mcpu=tahiti -o /dev/null %s 2>&1 | FileCheck -check-prefix=FAIL %s
; RUN: not --crash llc -march=amdgcn -mcpu=hawaii -o /dev/null %s 2>&1 | FileCheck -check-prefix=FAIL %s
; RUN: not --crash llc -march=amdgcn -mcpu=fiji -o /dev/null %s 2>&1 | FileCheck -check-prefix=FAIL %s
; RUN: not --crash llc -march=amdgcn -mcpu=gfx900 -o /dev/null %s 2>&1 | FileCheck -check-prefix=FAIL %s
; RUN: not --crash llc -march=amdgcn -mcpu=gfx1010 -o /dev/null %s 2>&1 | FileCheck -check-prefix=FAIL %s

; Make sure selection of these intrinsics fails on targets that do not
; have the instruction available.
; FIXME: Should also really make sure the v2f16 version fails.

; FAIL: LLVM ERROR: Cannot select: {{.+}}: f32,ch = BUFFER_ATOMIC_FADD
define amdgpu_cs void @atomic_fadd(<4 x i32> inreg %arg0) {
  %ret = call float @llvm.amdgcn.buffer.atomic.fadd.f32(float 1.0, <4 x i32> %arg0, i32 0, i32 112, i1 false)
  ret void
}

declare float @llvm.amdgcn.buffer.atomic.fadd.f32(float, <4 x i32>, i32, i32, i1 immarg) #0

attributes #0 = { nounwind }
