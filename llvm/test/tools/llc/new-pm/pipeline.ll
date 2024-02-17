; RUN: llc -mtriple=x86_64-pc-linux-gnu -enable-new-pm -print-pipeline-passes -filetype=null %s | FileCheck %s

; CHECK: require<profile-summary>,require<collector-metadata>
; CHECK: MachineSanitizerBinaryMetadata,FreeMachineFunctionPass

