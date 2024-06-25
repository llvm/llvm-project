; RUN: llc -mtriple=x86_64-pc-linux-gnu -enable-new-pm -print-pipeline-passes -start-before=mergeicmps -stop-after=gc-lowering -filetype=null %s | FileCheck --match-full-lines %s --check-prefix=NULL
; RUN: llc -mtriple=x86_64-pc-linux-gnu -enable-new-pm -print-pipeline-passes -start-before=mergeicmps -stop-after=gc-lowering -o /dev/null %s | FileCheck --match-full-lines %s --check-prefix=OBJ

; NULL: require<MachineModuleAnalysis>,require<profile-summary>,require<collector-metadata>,function(verify,loop-mssa(loop-reduce),mergeicmps,expand-memcmp,gc-lowering,ee-instrument<post-inline>,verify)
; OBJ: require<MachineModuleAnalysis>,require<profile-summary>,require<collector-metadata>,function(verify,loop-mssa(loop-reduce),mergeicmps,expand-memcmp,gc-lowering,ee-instrument<post-inline>,verify),PrintMIRPreparePass,function(machine-function(print),invalidate<machine-function-info>)
