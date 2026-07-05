; RUN: not llc -mtriple aarch64-none-linux-gnu -filetype=null %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: streaming SVE functions require SME
define void @streaming(i64 noundef %n) "aarch64_pstate_sm_enabled" nounwind {
entry:
  ret void
}
