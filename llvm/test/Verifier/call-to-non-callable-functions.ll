; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare amdgpu_cs_chain void @callee_amdgpu_cs_chain()

; CHECK: calling convention does not permit calls
; CHECK-NEXT: call amdgpu_cs_chain void @callee_amdgpu_cs_chain()
define amdgpu_cs_chain void @call_caller_amdgpu_cs_chain() {
entry:
  call amdgpu_cs_chain void @callee_amdgpu_cs_chain()
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK-NEXT: call amdgpu_cs_chain void %func()
define amdgpu_cs_chain void @indirect_call_caller_amdgpu_cs_chain(ptr %func) {
entry:
  call amdgpu_cs_chain void %func()
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK-NEXT: invoke amdgpu_cs_chain void @callee_amdgpu_cs_chain()
; CHECK-NEXT: to label %cont unwind label %unwind
define amdgpu_cs_chain void @invoke_caller_amdgpu_cs_chain() {
entry:
  invoke amdgpu_cs_chain void @callee_amdgpu_cs_chain() to label %cont unwind label %unwind
  ret void

cont:
  ret void

unwind:
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK: invoke amdgpu_cs_chain void %func()
; CHECK-NEXT: to label %cont unwind label %unwind
define amdgpu_cs_chain void @indirect_invoke_caller_amdgpu_cs_chain(ptr %func) {
entry:
  invoke amdgpu_cs_chain void %func() to label %cont unwind label %unwind
  ret void

cont:
  ret void

unwind:
  ret void
}

declare amdgpu_cs_chain_preserve void @callee_amdgpu_cs_chain_preserve()

; CHECK: calling convention does not permit calls
; CHECK-NEXT: call amdgpu_cs_chain_preserve void @callee_amdgpu_cs_chain_preserve()
define amdgpu_cs_chain_preserve void @call_caller_amdgpu_cs_chain_preserve() {
entry:
  call amdgpu_cs_chain_preserve void @callee_amdgpu_cs_chain_preserve()
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK-NEXT: call amdgpu_cs_chain_preserve void %func()
define amdgpu_cs_chain_preserve void @indirect_call_caller_amdgpu_cs_chain_preserve(ptr %func) {
entry:
  call amdgpu_cs_chain_preserve void %func()
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK-NEXT: invoke amdgpu_cs_chain_preserve void @callee_amdgpu_cs_chain_preserve()
; CHECK-NEXT: to label %cont unwind label %unwind
define amdgpu_cs_chain_preserve void @invoke_caller_amdgpu_cs_chain_preserve() {
entry:
  invoke amdgpu_cs_chain_preserve void @callee_amdgpu_cs_chain_preserve() to label %cont unwind label %unwind
  ret void

cont:
  ret void

unwind:
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK: invoke amdgpu_cs_chain_preserve void %func()
; CHECK-NEXT: to label %cont unwind label %unwind
define amdgpu_cs_chain_preserve void @indirect_invoke_caller_amdgpu_cs_chain_preserve(ptr %func) {
entry:
  invoke amdgpu_cs_chain_preserve void %func() to label %cont unwind label %unwind
  ret void

cont:
  ret void

unwind:
  ret void
}

declare amdgpu_cs void @callee_amdgpu_cs()

; CHECK: calling convention does not permit calls
; CHECK-NEXT: call amdgpu_cs void @callee_amdgpu_cs()
define amdgpu_cs void @call_caller_amdgpu_cs() {
entry:
  call amdgpu_cs void @callee_amdgpu_cs()
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK-NEXT: call amdgpu_cs void %func()
define amdgpu_cs void @indirect_call_caller_amdgpu_cs(ptr %func) {
entry:
  call amdgpu_cs void %func()
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK-NEXT: invoke amdgpu_cs void @callee_amdgpu_cs()
; CHECK-NEXT: to label %cont unwind label %unwind
define amdgpu_cs void @invoke_caller_amdgpu_cs() {
entry:
  invoke amdgpu_cs void @callee_amdgpu_cs() to label %cont unwind label %unwind
  ret void

cont:
  ret void

unwind:
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK: invoke amdgpu_cs void %func()
; CHECK-NEXT: to label %cont unwind label %unwind
define amdgpu_cs void @indirect_invoke_caller_amdgpu_cs(ptr %func) {
entry:
  invoke amdgpu_cs void %func() to label %cont unwind label %unwind
  ret void

cont:
  ret void

unwind:
  ret void
}

declare amdgpu_es void @callee_amdgpu_es()

; CHECK: calling convention does not permit calls
; CHECK-NEXT: call amdgpu_es void @callee_amdgpu_es()
define amdgpu_es void @call_caller_amdgpu_es() {
entry:
  call amdgpu_es void @callee_amdgpu_es()
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK-NEXT: call amdgpu_es void %func()
define amdgpu_es void @indirect_call_caller_amdgpu_es(ptr %func) {
entry:
  call amdgpu_es void %func()
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK-NEXT: invoke amdgpu_es void @callee_amdgpu_es()
; CHECK-NEXT: to label %cont unwind label %unwind
define amdgpu_es void @invoke_caller_amdgpu_es() {
entry:
  invoke amdgpu_es void @callee_amdgpu_es() to label %cont unwind label %unwind
  ret void

cont:
  ret void

unwind:
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK: invoke amdgpu_es void %func()
; CHECK-NEXT: to label %cont unwind label %unwind
define amdgpu_es void @indirect_invoke_caller_amdgpu_es(ptr %func) {
entry:
  invoke amdgpu_es void %func() to label %cont unwind label %unwind
  ret void

cont:
  ret void

unwind:
  ret void
}

declare amdgpu_gs void @callee_amdgpu_gs()

; CHECK: calling convention does not permit calls
; CHECK-NEXT: call amdgpu_gs void @callee_amdgpu_gs()
define amdgpu_gs void @call_caller_amdgpu_gs() {
entry:
  call amdgpu_gs void @callee_amdgpu_gs()
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK-NEXT: call amdgpu_gs void %func()
define amdgpu_gs void @indirect_call_caller_amdgpu_gs(ptr %func) {
entry:
  call amdgpu_gs void %func()
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK-NEXT: invoke amdgpu_gs void @callee_amdgpu_gs()
; CHECK-NEXT: to label %cont unwind label %unwind
define amdgpu_gs void @invoke_caller_amdgpu_gs() {
entry:
  invoke amdgpu_gs void @callee_amdgpu_gs() to label %cont unwind label %unwind
  ret void

cont:
  ret void

unwind:
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK: invoke amdgpu_gs void %func()
; CHECK-NEXT: to label %cont unwind label %unwind
define amdgpu_gs void @indirect_invoke_caller_amdgpu_gs(ptr %func) {
entry:
  invoke amdgpu_gs void %func() to label %cont unwind label %unwind
  ret void

cont:
  ret void

unwind:
  ret void
}

declare amdgpu_hs void @callee_amdgpu_hs()

; CHECK: calling convention does not permit calls
; CHECK-NEXT: call amdgpu_hs void @callee_amdgpu_hs()
define amdgpu_hs void @call_caller_amdgpu_hs() {
entry:
  call amdgpu_hs void @callee_amdgpu_hs()
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK-NEXT: call amdgpu_hs void %func()
define amdgpu_hs void @indirect_call_caller_amdgpu_hs(ptr %func) {
entry:
  call amdgpu_hs void %func()
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK-NEXT: invoke amdgpu_hs void @callee_amdgpu_hs()
; CHECK-NEXT: to label %cont unwind label %unwind
define amdgpu_hs void @invoke_caller_amdgpu_hs() {
entry:
  invoke amdgpu_hs void @callee_amdgpu_hs() to label %cont unwind label %unwind
  ret void

cont:
  ret void

unwind:
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK: invoke amdgpu_hs void %func()
; CHECK-NEXT: to label %cont unwind label %unwind
define amdgpu_hs void @indirect_invoke_caller_amdgpu_hs(ptr %func) {
entry:
  invoke amdgpu_hs void %func() to label %cont unwind label %unwind
  ret void

cont:
  ret void

unwind:
  ret void
}

declare amdgpu_kernel void @callee_amdgpu_kernel()

; CHECK: calling convention does not permit calls
; CHECK-NEXT: call amdgpu_kernel void @callee_amdgpu_kernel()
define amdgpu_kernel void @call_caller_amdgpu_kernel() {
entry:
  call amdgpu_kernel void @callee_amdgpu_kernel()
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK-NEXT: call amdgpu_kernel void %func()
define amdgpu_kernel void @indirect_call_caller_amdgpu_kernel(ptr %func) {
entry:
  call amdgpu_kernel void %func()
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK-NEXT: invoke amdgpu_kernel void @callee_amdgpu_kernel()
; CHECK-NEXT: to label %cont unwind label %unwind
define amdgpu_kernel void @invoke_caller_amdgpu_kernel() {
entry:
  invoke amdgpu_kernel void @callee_amdgpu_kernel() to label %cont unwind label %unwind
  ret void

cont:
  ret void

unwind:
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK: invoke amdgpu_kernel void %func()
; CHECK-NEXT: to label %cont unwind label %unwind
define amdgpu_kernel void @indirect_invoke_caller_amdgpu_kernel(ptr %func) {
entry:
  invoke amdgpu_kernel void %func() to label %cont unwind label %unwind
  ret void

cont:
  ret void

unwind:
  ret void
}

declare amdgpu_ls void @callee_amdgpu_ls()

; CHECK: calling convention does not permit calls
; CHECK-NEXT: call amdgpu_ls void @callee_amdgpu_ls()
define amdgpu_ls void @call_caller_amdgpu_ls() {
entry:
  call amdgpu_ls void @callee_amdgpu_ls()
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK-NEXT: call amdgpu_ls void %func()
define amdgpu_ls void @indirect_call_caller_amdgpu_ls(ptr %func) {
entry:
  call amdgpu_ls void %func()
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK-NEXT: invoke amdgpu_ls void @callee_amdgpu_ls()
; CHECK-NEXT: to label %cont unwind label %unwind
define amdgpu_ls void @invoke_caller_amdgpu_ls() {
entry:
  invoke amdgpu_ls void @callee_amdgpu_ls() to label %cont unwind label %unwind
  ret void

cont:
  ret void

unwind:
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK: invoke amdgpu_ls void %func()
; CHECK-NEXT: to label %cont unwind label %unwind
define amdgpu_ls void @indirect_invoke_caller_amdgpu_ls(ptr %func) {
entry:
  invoke amdgpu_ls void %func() to label %cont unwind label %unwind
  ret void

cont:
  ret void

unwind:
  ret void
}

declare amdgpu_ps void @callee_amdgpu_ps()

; CHECK: calling convention does not permit calls
; CHECK-NEXT: call amdgpu_ps void @callee_amdgpu_ps()
define amdgpu_ps void @call_caller_amdgpu_ps() {
entry:
  call amdgpu_ps void @callee_amdgpu_ps()
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK-NEXT: call amdgpu_ps void %func()
define amdgpu_ps void @indirect_call_caller_amdgpu_ps(ptr %func) {
entry:
  call amdgpu_ps void %func()
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK-NEXT: invoke amdgpu_ps void @callee_amdgpu_ps()
; CHECK-NEXT: to label %cont unwind label %unwind
define amdgpu_ps void @invoke_caller_amdgpu_ps() {
entry:
  invoke amdgpu_ps void @callee_amdgpu_ps() to label %cont unwind label %unwind
  ret void

cont:
  ret void

unwind:
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK: invoke amdgpu_ps void %func()
; CHECK-NEXT: to label %cont unwind label %unwind
define amdgpu_ps void @indirect_invoke_caller_amdgpu_ps(ptr %func) {
entry:
  invoke amdgpu_ps void %func() to label %cont unwind label %unwind
  ret void

cont:
  ret void

unwind:
  ret void
}

declare amdgpu_vs void @callee_amdgpu_vs()

; CHECK: calling convention does not permit calls
; CHECK-NEXT: call amdgpu_vs void @callee_amdgpu_vs()
define amdgpu_vs void @call_caller_amdgpu_vs() {
entry:
  call amdgpu_vs void @callee_amdgpu_vs()
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK-NEXT: call amdgpu_vs void %func()
define amdgpu_vs void @indirect_call_caller_amdgpu_vs(ptr %func) {
entry:
  call amdgpu_vs void %func()
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK-NEXT: invoke amdgpu_vs void @callee_amdgpu_vs()
; CHECK-NEXT: to label %cont unwind label %unwind
define amdgpu_vs void @invoke_caller_amdgpu_vs() {
entry:
  invoke amdgpu_vs void @callee_amdgpu_vs() to label %cont unwind label %unwind
  ret void

cont:
  ret void

unwind:
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK: invoke amdgpu_vs void %func()
; CHECK-NEXT: to label %cont unwind label %unwind
define amdgpu_vs void @indirect_invoke_caller_amdgpu_vs(ptr %func) {
entry:
  invoke amdgpu_vs void %func() to label %cont unwind label %unwind
  ret void

cont:
  ret void

unwind:
  ret void
}

declare spir_kernel void @callee_spir_kernel()

; CHECK: calling convention does not permit calls
; CHECK-NEXT: call spir_kernel void @callee_spir_kernel()
define spir_kernel void @call_caller_spir_kernel() {
entry:
  call spir_kernel void @callee_spir_kernel()
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK-NEXT: call spir_kernel void %func()
define spir_kernel void @indirect_call_caller_spir_kernel(ptr %func) {
entry:
  call spir_kernel void %func()
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK-NEXT: invoke spir_kernel void @callee_spir_kernel()
; CHECK-NEXT: to label %cont unwind label %unwind
define spir_kernel void @invoke_caller_spir_kernel() {
entry:
  invoke spir_kernel void @callee_spir_kernel() to label %cont unwind label %unwind
  ret void

cont:
  ret void

unwind:
  ret void
}

; CHECK: calling convention does not permit calls
; CHECK: invoke spir_kernel void %func()
; CHECK-NEXT: to label %cont unwind label %unwind
define spir_kernel void @indirect_invoke_caller_spir_kernel(ptr %func) {
entry:
  invoke spir_kernel void %func() to label %cont unwind label %unwind
  ret void

cont:
  ret void

unwind:
  ret void
}
