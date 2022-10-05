; Test that llvm-reduce can remove uninteresting metadata from an IR file.
; The Metadata pass erases named & unnamed metadata nodes.
;
; RUN: llvm-reduce %s -o %t --delta-passes=metadata --test FileCheck --test-arg %s --test-arg --input-file
; RUN: FileCheck %s < %t

@global = global i32 0, !dbg !0

define void @main() !dbg !0 {
   ret void, !dbg !0
}

!uninteresting = !{!0}
; CHECK: !interesting = !{![[I:[0-9]+]]}
!interesting = !{!1}

!0 = !{!"uninteresting"}
; CHECK: ![[I]] = !{!"interesting"}
!1 = !{!"interesting"}
