; RUN: llc < %s -mtriple=i686-- -tailcallopt=false | FileCheck %s

	%struct.foo = type { [4 x i32] }

define fastcc void @bar(ptr noalias sret(%struct.foo) %agg.result) nounwind  {
entry:
	store i32 1, ptr %agg.result, align 8
        ret void
}
; CHECK: bar
; CHECK: ret{{[^4]*$}}

@dst = external dso_local global i32

define void @foo() nounwind {
	%memtmp = alloca %struct.foo, align 4
        call fastcc void @bar(ptr sret(%struct.foo) %memtmp ) nounwind
        %tmp6 = load i32, ptr %memtmp
        store i32 %tmp6, ptr @dst
        ret void
}
; CHECK: foo
; CHECK: ret{{[^4]*$}}
