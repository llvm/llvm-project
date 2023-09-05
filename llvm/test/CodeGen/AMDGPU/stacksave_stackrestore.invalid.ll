; RUN: split-file %s %t
; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn-amd-amdpal -mcpu=gfx1030 -filetype=null %t/stacksave-error.ll 2>&1 | FileCheck -check-prefix=ERR-SAVE-SDAG %s
; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn-amd-amdpal -mcpu=gfx1030 -filetype=null %t/stackrestore-error.ll 2>&1 | FileCheck -check-prefix=ERR-RESTORE-SDAG %s

; RUN: not --crash llc -global-isel=1 -mtriple=amdgcn-amd-amdpal -mcpu=gfx1030 -filetype=null %t/stacksave-error.ll 2>&1 | FileCheck -check-prefix=ERR-SAVE-GISEL %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgcn-amd-amdpal -mcpu=gfx1030 -filetype=null %t/stackrestore-error.ll 2>&1 | FileCheck -check-prefix=ERR-RESTORE-GISEL %s

; Test that an error is produced if stacksave/stackrestore are used
; with the wrong (default) address space.

;--- stacksave-error.ll

declare ptr @llvm.stacksave.p0()

; ERR-SAVE-SDAG: LLVM ERROR: Cannot select: {{.+}}: i64,ch = stacksave
; ERR-SAVE-GISEL: LLVM ERROR: unable to legalize instruction: %{{[0-9]+}}:_(p0) = G_STACKSAVE (in function: func_store_stacksave)
define void @func_store_stacksave() {
  %stacksave = call ptr @llvm.stacksave.p0()
  call void asm sideeffect "; use $0", "s"(ptr %stacksave)
  ret void
}

;--- stackrestore-error.ll

declare void @llvm.stackrestore.p0(ptr)

; ERR-RESTORE-SDAG: LLVM ERROR: Cannot select: {{.+}}: ch = stackrestore {{.+}}, {{.+}}
; ERR-RESTORE-GISEL: LLVM ERROR: unable to legalize instruction: G_STACKRESTORE %{{[0-9]+}}:_(p0) (in function: func_stacksave_sgpr)
define amdgpu_gfx void @func_stacksave_sgpr(ptr inreg %stack) {
  call void @llvm.stackrestore.p0(ptr %stack)
  ret void
}
