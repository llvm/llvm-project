; RUN: llc -mtriple=wasm32 -mattr=+reference-types -global-isel -global-isel-abort=2 -pass-remarks-missed='gisel*' %s -o - 2>&1 | FileCheck %s
; RUN: llc -mtriple=wasm64 -mattr=+reference-types -global-isel -global-isel-abort=2 -pass-remarks-missed='gisel*' %s -o - 2>&1 | FileCheck %s

; GlobalISel does not yet correctly model WebAssembly reference types
; (externref/funcref): the generic getLLTForType derives an integer scalar LLT
; from their pointer layout type, which would round-trip references through
; invalid integer loads/bitcasts. Until that is fixed, the WebAssembly
; CallLowering bails out for any function passing or returning a reference type,
; so instruction selection falls back to SelectionDAG. This test verifies that
; fallback (rather than emission of incorrect code).

define target("wasm.externref") @ret_externref(target("wasm.externref") %a) {
; CHECK: remark: {{.*}} unable to lower arguments{{.*}}wasm.externref
; CHECK-LABEL: warning: Instruction selection used fallback path for ret_externref
  ret target("wasm.externref") %a
}

define target("wasm.funcref") @ret_funcref(target("wasm.funcref") %a) {
; CHECK: remark: {{.*}} unable to lower arguments{{.*}}wasm.funcref
; CHECK-LABEL: warning: Instruction selection used fallback path for ret_funcref
  ret target("wasm.funcref") %a
}

declare void @take_externref(target("wasm.externref"))

define void @call_take_externref(target("wasm.externref") %a) {
; CHECK: remark: {{.*}} unable to lower arguments{{.*}}wasm.externref
; CHECK-LABEL: warning: Instruction selection used fallback path for call_take_externref
  call void @take_externref(target("wasm.externref") %a)
  ret void
}

declare target("wasm.externref") @produce_externref()

define void @call_produce_externref() {
; CHECK: remark: {{.*}} unable to {{.*}}
; CHECK-LABEL: warning: Instruction selection used fallback path for call_produce_externref
  %ref = call target("wasm.externref") @produce_externref()
  ret void
}
