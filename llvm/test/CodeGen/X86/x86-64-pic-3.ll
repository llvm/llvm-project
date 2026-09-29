; RUN: llc < %s -mtriple=x86_64-pc-linux -relocation-model=pic | FileCheck %s


; CHECK-NOT: {{callq	f@PLT}}
; CHECK: {{callq	f}}
; CHECK-NOT: {{callq	f@PLT}}

define void @g() {
entry:
	call void @f( )
	ret void
}

define internal void @f() {
entry:
	ret void
}
