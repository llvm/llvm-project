; RUN: split-file %s %t
; RUN: llvm-as < %t/gdb.ll | llvm-dis | FileCheck -check-prefix=GDB %s
; RUN: llvm-as < %t/lldb.ll | llvm-dis | FileCheck -check-prefix=LLDB %s
; RUN: llvm-as < %t/sce.ll | llvm-dis | FileCheck -check-prefix=SCE %s
; RUN: llvm-as < %t/dbx.ll | llvm-dis | FileCheck -check-prefix=DBX %s
; RUN: llvm-as < %t/unknown.ll | llvm-dis | FileCheck -check-prefix=UNKNOWN %s

;--- gdb.ll
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"debugger-tuning", !"gdb"}
; GDB: !0 = !{i32 2, !"debugger-tuning", !"gdb"}

;--- lldb.ll
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"debugger-tuning", !"lldb"}
; LLDB: !0 = !{i32 2, !"debugger-tuning", !"lldb"}

;--- sce.ll
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"debugger-tuning", !"sce"}
; SCE: !0 = !{i32 2, !"debugger-tuning", !"sce"}

;--- dbx.ll
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"debugger-tuning", !"dbx"}
; DBX: !0 = !{i32 2, !"debugger-tuning", !"dbx"}

;--- unknown.ll
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"debugger-tuning", !"arstarstarst"}
; UNKNOWN: !0 = !{i32 2, !"debugger-tuning", !"arstarstarst"}
