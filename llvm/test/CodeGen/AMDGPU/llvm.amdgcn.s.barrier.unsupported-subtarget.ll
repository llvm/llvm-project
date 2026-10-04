; RUN: split-file %s %t

; RUN: not llc -global-isel=0 -mtriple=amdgpu9.42-amd-amdhsa -filetype=null %t/signal.ll 2>&1 | FileCheck -check-prefix=ERR-SIGNAL %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu9.42-amd-amdhsa -filetype=null %t/signal-var.ll 2>&1 | FileCheck -check-prefix=ERR-SIGNAL-VAR %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu9.42-amd-amdhsa -filetype=null %t/signal-isfirst.ll 2>&1 | FileCheck -check-prefix=ERR-SIGNAL-ISFIRST %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu9.42-amd-amdhsa -filetype=null %t/init.ll 2>&1 | FileCheck -check-prefix=ERR-INIT %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu9.42-amd-amdhsa -filetype=null %t/join.ll 2>&1 | FileCheck -check-prefix=ERR-JOIN %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu9.42-amd-amdhsa -filetype=null %t/wait.ll 2>&1 | FileCheck -check-prefix=ERR-WAIT %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu9.42-amd-amdhsa -filetype=null %t/leave.ll 2>&1 | FileCheck -check-prefix=ERR-LEAVE %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu9.42-amd-amdhsa -filetype=null %t/get-state.ll 2>&1 | FileCheck -check-prefix=ERR-GET-STATE %s

; TODO: GIsel should not crash.
; RUN: not --crash llc -global-isel=1 -mtriple=amdgpu9.42-amd-amdhsa -filetype=null %t/signal.ll 2>&1 | FileCheck -check-prefix=ERR-SIGNAL %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgpu9.42-amd-amdhsa -filetype=null %t/signal-var.ll 2>&1 | FileCheck -check-prefix=ERR-SIGNAL-VAR %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgpu9.42-amd-amdhsa -filetype=null %t/signal-isfirst.ll 2>&1 | FileCheck -check-prefix=ERR-SIGNAL-ISFIRST %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgpu9.42-amd-amdhsa -filetype=null %t/init.ll 2>&1 | FileCheck -check-prefix=ERR-INIT %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgpu9.42-amd-amdhsa -filetype=null %t/join.ll 2>&1 | FileCheck -check-prefix=ERR-JOIN %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgpu9.42-amd-amdhsa -filetype=null %t/wait.ll 2>&1 | FileCheck -check-prefix=ERR-WAIT %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu9.42-amd-amdhsa -filetype=null %t/leave.ll 2>&1 | FileCheck -check-prefix=ERR-LEAVE %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgpu9.42-amd-amdhsa -filetype=null %t/get-state.ll 2>&1 | FileCheck -check-prefix=ERR-GET-STATE %s

;--- signal.ll
; ERR-SIGNAL: error: {{.*}}in function @signal void (): llvm.amdgcn.s.barrier.signal requires target feature 'gfx12-insts'

define amdgpu_kernel void @signal() {
  call void @llvm.amdgcn.s.barrier.signal(i32 -1)
  ret void
}

;--- signal-var.ll
; ERR-SIGNAL-VAR: error: {{.*}}in function @signal_var void (): llvm.amdgcn.s.barrier.signal.var requires target feature 'gfx12-insts'

@bar_signal_var = addrspace(15) global target("amdgcn.named.barrier", 0) poison

define amdgpu_kernel void @signal_var() {
  call void @llvm.amdgcn.s.barrier.signal.var(ptr addrspace(15) @bar_signal_var, i32 4)
  ret void
}

;--- signal-isfirst.ll
; ERR-SIGNAL-ISFIRST: error: {{.*}}in function @signal_isfirst void (): llvm.amdgcn.s.barrier.signal.isfirst requires target feature 'gfx12-insts'

define amdgpu_kernel void @signal_isfirst() {
  %r = call i1 @llvm.amdgcn.s.barrier.signal.isfirst(i32 -1)
  ret void
}

;--- init.ll
; ERR-INIT: error: {{.*}}in function @init void (): llvm.amdgcn.s.barrier.init requires target feature 'gfx12-insts'

@bar_init = addrspace(15) global target("amdgcn.named.barrier", 0) poison

define amdgpu_kernel void @init() {
  call void @llvm.amdgcn.s.barrier.init(ptr addrspace(15) @bar_init, i32 4)
  ret void
}

;--- join.ll
; ERR-JOIN: error: {{.*}}in function @join void (): llvm.amdgcn.s.barrier.join requires target feature 'gfx12-insts'

@bar_join = addrspace(15) global target("amdgcn.named.barrier", 0) poison

define amdgpu_kernel void @join() {
  call void @llvm.amdgcn.s.barrier.join(ptr addrspace(15) @bar_join)
  ret void
}

;--- wait.ll
; ERR-WAIT: error: {{.*}}in function @wait void (): llvm.amdgcn.s.barrier.wait requires target feature 'gfx12-insts'

define amdgpu_kernel void @wait() {
  call void @llvm.amdgcn.s.barrier.wait(i16 -1)
  ret void
}

;--- leave.ll
; ERR-LEAVE: error: {{.*}}in function @leave void (): llvm.amdgcn.s.barrier.leave requires target feature 'gfx12-insts'

define amdgpu_kernel void @leave() {
  call void @llvm.amdgcn.s.barrier.leave(i16 -1)
  ret void
}

;--- get-state.ll
; ERR-GET-STATE: error: {{.*}}in function @get_state void (): llvm.amdgcn.s.get.barrier.state requires target feature 'gfx12-insts'

define amdgpu_kernel void @get_state() {
  %r = call i32 @llvm.amdgcn.s.get.barrier.state(i32 -1)
  ret void
}
