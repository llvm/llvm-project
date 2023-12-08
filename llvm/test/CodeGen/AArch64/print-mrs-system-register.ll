; RUN: llc -mtriple=arm64-apple-darwin %s -o - | FileCheck %s

; CHECK: mrs x0, CPM_IOACC_CTL_EL3

define void @foo1() #0 {
entry:
  tail call void asm sideeffect "mrs x0, cpm_ioacc_ctl_el3", ""()
  ret void
}

attributes #0 = { "target-cpu"="cyclone" }
