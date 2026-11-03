; RUN: llc -mtriple=x86_64-pc-linux-gnu -enable-new-pm -print-pipeline-passes -start-before=gc-lowering -stop-after=gc-lowering -filetype=null %s | FileCheck --match-full-lines %s --check-prefix=NULL
; RUN: llc -mtriple=x86_64-pc-linux-gnu -enable-new-pm -print-pipeline-passes -start-before=gc-lowering -stop-after=gc-lowering -o /dev/null %s | FileCheck --match-full-lines %s --check-prefix=OBJ
; RUN: llc -mtriple=x86_64-pc-linux-gnu -enable-new-pm -start-before=gc-lowering -stop-after=gc-lowering -verify-machineinstrs -o - %s | FileCheck %s --check-prefix=MIR

; NULL: require<MachineModuleAnalysis>,require<profile-summary>,require<collector-metadata>,require<runtime-libcall-info>,require<libcall-lowering-info>,function(verify,gc-lowering,verify),function(machine-function(verify),free-machine-function)
; OBJ: require<MachineModuleAnalysis>,require<profile-summary>,require<collector-metadata>,require<runtime-libcall-info>,require<libcall-lowering-info>,function(verify,gc-lowering,verify),function(machine-function(verify)),PrintMIRPass

; Only @f is printed, even though no machine pass created its MF.
; MIR-NOT: {{^}}name:
; MIR: {{^}}name: f{{$}}
; MIR-NOT: {{^}}name:

declare void @ext()

define available_externally void @ae() {
  ret void
}

define void @f() {
  call void @ext()
  ret void
}
