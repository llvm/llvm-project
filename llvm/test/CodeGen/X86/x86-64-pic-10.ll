; RUN: llc < %s -mtriple=x86_64-pc-linux -relocation-model=pic -o %t1
; RUN: grep "callq	g@PLT" %t1

@g = alias weak i32 ()* @f

define void @h() {
entry:
	%tmp31 = call i32 @g()
        ret void
}

declare extern_weak i32 @f()
