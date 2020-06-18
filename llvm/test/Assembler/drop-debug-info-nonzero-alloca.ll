; RUN: llvm-as < %s -o %t.bc -data-layout=A5 2>&1 | FileCheck -check-prefixes=ALL,AS %s
; RUN: llvm-dis < %t.bc | FileCheck -check-prefixes=ALL,DIS %s
; RUN: opt < %s -S -data-layout=A5 2>&1 | FileCheck -check-prefixes=ALL,AS %s
; RUN: opt < %t.bc -S | FileCheck -check-prefixes=ALL,DIS %s

define void @foo() {
entry:
; DIS: target datalayout = "A5"
; DIS: %tmp = alloca i32, align 4, addrspace(5)
  %tmp = alloca i32, addrspace(5)
  call void @llvm.dbg.value(
      metadata i8* undef,
      metadata !DILocalVariable(scope: !1),
      metadata !DIExpression())
; ALL-NOT: Allocation instruction pointer not in the stack address space!
; AS: llvm.dbg.value intrinsic requires a !dbg attachment
; AS: warning: ignoring invalid debug info in <stdin>
ret void
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DISubprogram(name: "foo")
