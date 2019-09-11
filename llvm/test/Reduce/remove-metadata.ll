; Test that llvm-reduce can remove uninteresting metadata from an IR file.
; The Metadata pass erases named & unnamed metadata nodes.
;
; RUN: rm -rf %t
; RUN: mkdir %t
; copy the test file to preserve executable bit
; RUN: cp %p/Inputs/remove-metadata.py %t/test.py
; get the python path from lit
; RUN: echo "#!" %python > %t/test.py
; then include the rest of the test script
; RUN: cat %p/Inputs/remove-metadata.py >> %t/test.py

; RUN: llvm-reduce --test %t/test.py %s -o %t/out.ll
; RUN: cat %t/out.ll | FileCheck -implicit-check-not=! %s
; REQUIRES: plugins

@global = global i32 0, !dbg !0

define void @main() !dbg !0 {
   ret void, !dbg !0
}

!uninteresting = !{!0}
; CHECK: !interesting = !{!0}
!interesting = !{!1}

!0 = !{!"uninteresting"}
; CHECK: !0 = !{!"interesting"}
!1 = !{!"interesting"}
