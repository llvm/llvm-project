; RUN: llc -mtriple=x86_64-pc-linux-gnu -enable-new-pm -print-pipeline-passes=tree -start-before=gc-lowering -stop-after=gc-lowering -filetype=null %s | FileCheck --match-full-lines %s --check-prefix=NULL
; RUN: llc -mtriple=x86_64-pc-linux-gnu -enable-new-pm -print-pipeline-passes=tree -start-before=gc-lowering -stop-after=gc-lowering -o /dev/null %s | FileCheck --match-full-lines %s --check-prefix=OBJ

; NULL:      require<MachineModuleAnalysis>
; NULL-NEXT: require<profile-summary>
; NULL-NEXT: require<collector-metadata>
; NULL-NEXT: require<runtime-libcall-info>
; NULL-NEXT: require<libcall-lowering-info>
; NULL-NEXT: function
; NULL-NEXT:   verify
; NULL-NEXT:   gc-lowering
; NULL-NEXT:   verify
; NULL-NEXT:   free-machine-function

; OBJ:      require<MachineModuleAnalysis>
; OBJ-NEXT: require<profile-summary>
; OBJ-NEXT: require<collector-metadata>
; OBJ-NEXT: require<runtime-libcall-info>
; OBJ-NEXT: require<libcall-lowering-info>
; OBJ-NEXT: function
; OBJ-NEXT:   verify
; OBJ-NEXT:   gc-lowering
; OBJ-NEXT:   verify
; OBJ-NEXT: PrintMIRPreparePass
; OBJ-NEXT: function
; OBJ-NEXT:   machine-function
; OBJ-NEXT:     verify
; OBJ-NEXT:     print
; OBJ-NEXT:   free-machine-function
