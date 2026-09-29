; RUN: not llc -mcpu=sm_100a -mtriple=nvptx64 -mattr=+ptx86 %s -o /dev/null 2>&1 | FileCheck %s

; Test that we get a clear error message when using an unsupported syncscope.

; CHECK: NVPTX backend does not support syncscope "agent"
; CHECK: Supported syncscopes are: singlethread, <empty string>, block, cluster, device
define i32 @cmpxchg_unsupported_syncscope_agent(ptr %addr, i32 %cmp, i32 %new) {
  %result = cmpxchg ptr %addr, i32 %cmp, i32 %new syncscope("agent") monotonic monotonic
  %value = extractvalue { i32, i1 } %result, 0
  ret i32 %value
}
