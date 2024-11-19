; RUN: llc -mtriple=x86_64-pc-linux-gnu -enable-new-pm -print-pipeline-passes -start-before=mergeicmps -stop-after=gc-lowering -filetype=null %s | FileCheck --match-full-lines %s --check-prefix=NULL
; RUN: llc -mtriple=x86_64-pc-linux-gnu -enable-new-pm -print-pipeline-passes -start-before=mergeicmps -stop-after=gc-lowering -o /dev/null %s | FileCheck --match-full-lines %s --check-prefix=OBJ

; NULL: verify,require<MachineModuleAnalysis>,require<profile-summary>,require<collector-metadata>,function(mergeicmps,expand-memcmp,gc-lowering),function(verify,machine-function(require<machine-loops>,verify),invalidate<machine-function-info>)
; OBJ: verify,require<MachineModuleAnalysis>,require<profile-summary>,require<collector-metadata>,function(mergeicmps,expand-memcmp,gc-lowering),function(verify),PrintMIRPreparePass,function(machine-function(require<machine-loops>,verify,print),invalidate<machine-function-info>)
