; RUN: split-file %s %t
; RUN: not --crash llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx1030 -filetype=null %t/stacksave-error.ll 2>&1 | FileCheck -check-prefix=ERR-SAVE %s
; RUN: not --crash llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx1030 -filetype=null %t/stackrestore-error.ll 2>&1 | FileCheck -check-prefix=ERR-RESTORE %s

; Test that an error is produced if stacksave/stackrestore are used
; with the wrong (default) address space.

;--- stacksave-error.ll

declare ptr @llvm.stacksave.p0()

; ERR-SAVE: LLVM ERROR: Cannot select: {{.+}}: i64,ch = stacksave
define void @func_store_stacksave() {
  %stacksave = call ptr @llvm.stacksave.p0()
  call void asm sideeffect "; use $0", "s"(ptr %stacksave)
  ret void
}

;--- stackrestore-error.ll

declare void @llvm.stackrestore.p0(ptr)

; ERR-RESTORE: LLVM ERROR: Cannot select: {{.+}}: ch = stackrestore {{.+}}, {{.+}}
define amdgpu_gfx void @func_stacksave_sgpr(ptr inreg %stack) {
  call void @llvm.stackrestore.p0(ptr %stack)
  ret void
}
