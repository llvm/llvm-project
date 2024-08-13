; Test that llvm-reduce can remove uninteresting metadata from an IR file.
; The Metadata pass erases named & unnamed metadata nodes.
;
; RUN: llvm-reduce --aggressive-md --test %python --test-arg %p/Inputs/remove-metadata.py %s -o %t
; RUN: FileCheck -implicit-check-not=! %s < %t 

@global = global i32 0, !dbg !0

define void @main() !dbg !0 {
   ret void, !dbg !0
}

!uninteresting = !{!0}
; CHECK: !interesting = !{}
!interesting = !{!1}

!0 = !{!"uninteresting"}
!1 = !{!"interesting"}
