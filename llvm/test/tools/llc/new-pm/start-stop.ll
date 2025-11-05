; RUN: llc -mtriple=x86_64-pc-linux-gnu -enable-new-pm -print-pipeline-passes -start-before=mergeicmps -stop-after=gc-lowering -filetype=null %s | FileCheck --match-full-lines %s --check-prefix=NULL
; RUN: llc -mtriple=x86_64-pc-linux-gnu -enable-new-pm -print-pipeline-passes -start-before=mergeicmps -stop-after=gc-lowering -o /dev/null %s | FileCheck --match-full-lines %s --check-prefix=OBJ

; NULL: require<MachineModuleAnalysis>,require<profile-summary>,require<collector-metadata>,require<runtime-libcall-info>,function(verify,mergeicmps,expand-memcmp,gc-lowering,verify)
; OBJ: require<MachineModuleAnalysis>,require<profile-summary>,require<collector-metadata>,require<runtime-libcall-info>,function(verify,mergeicmps,expand-memcmp,gc-lowering,verify),PrintMIRPreparePass,function(machine-function(print),free-machine-function)
