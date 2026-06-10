; RUN: opt -p 'no-op-module' %s --filetype=null --disable-pipeline-verification --print-pipeline-passes | \
; RUN: FileCheck %s --check-prefix=TEXT

; RUN: opt -p 'no-op-module,function(no-op-function)' %s --filetype=null \
; RUN: --disable-pipeline-verification --print-pipeline-passes=tree | \
; RUN: FileCheck %s --strict-whitespace --match-full-lines --check-prefix=TREE

; TEXT: no-op-module,verify

; TREE:no-op-module
; TREE-NEXT:function
; TREE-NEXT:  no-op-function
; TREE-NEXT:verify
