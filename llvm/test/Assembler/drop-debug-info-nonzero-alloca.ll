; RUN: llvm-as < %s -o %t.bc -data-layout=A5 2>&1 | FileCheck -check-prefixes=AS %s
; RUN: llvm-dis < %t.bc | FileCheck -check-prefixes=DIS %s
; RUN: opt < %s -S -data-layout=A5 2>&1 | FileCheck -check-prefixes=AS %s
; RUN: opt < %t.bc -S | FileCheck -check-prefixes=DIS %s

define void @foo() {
entry:
; DIS: target datalayout = "A5"
; DIS: %tmp = alloca i32, align 4, addrspace(5)
  %tmp = alloca i32, addrspace(5)
  call void @llvm.dbg.value(
      metadata ptr undef,
      metadata !DILocalVariable(scope: !1),
      metadata !DIExpression())
; AS: invalid #dbg record DILocation
; AS: #dbg_value(ptr undef, !{{[0-9]+}}, !DIExpression(), (null))
; AS: label %entry
; AS: ptr @foo
; AS: warning: ignoring invalid debug info in <stdin>

ret void
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DISubprogram(name: "foo")
