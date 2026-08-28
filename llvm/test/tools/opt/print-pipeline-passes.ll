; RUN: opt -p 'no-op-module' %s --filetype=null -S --print-pipeline-passes | \
; RUN: FileCheck %s --check-prefix=TEXT

; RUN: opt -p 'no-op-module,function(no-op-function)' %s --filetype=null \
; RUN: -S --print-pipeline-passes=tree | \
; RUN: FileCheck %s --strict-whitespace --match-full-lines --check-prefix=TREE

; TEXT: no-op-module,verify,print

; TREE:no-op-module
; TREE-NEXT:function
; TREE-NEXT:  no-op-function
; TREE-NEXT:verify
; TREE-NEXT:print
