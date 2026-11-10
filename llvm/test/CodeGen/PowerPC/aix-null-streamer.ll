; RUN: llc -mtriple=powerpc64-ibm-aix-xcoff -filetype=null %s

; Make sure the null streamer handles the XCOFF specific directives.

@local_bss = internal global i32 0
@d = global i32 4, !implicit.ref !0

define ptr @use_local_bss() {
  ret ptr @local_bss
}

define void @trap(i32 %a) {
  call void @llvm.ppc.trap(i32 %a), !annotation !1
  ret void
}

!0 = !{ptr @local_bss}
!1 = !{!"ppc-trap-reason", !"1", !"2"}

!llvm.commandline = !{!2}
!2 = !{!"clang -command -line"}
