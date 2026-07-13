; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -o /dev/null %s 2>&1 | FileCheck %s

; CHECK: warning: {{.*}}: unknown amdgcn-av metadata 'bogus'

define void @test_unknown_av() {
entry:
  fence seq_cst, !mmra !0
  ret void
}

!0 = !{!"amdgcn-av", !"bogus"}
