; RUN: llvm-as < %s | opt -std-compile-opts -o - | llc -o - | grep bork_directive | wc -l | grep 2

;; We don't want branch folding to fold asm directives.

define void @bork(i32 %param) {
entry:
	%tmp = icmp eq i32 %param, 0
        br i1 %tmp, label %cond_true, label %cond_false

cond_true:   
        call void asm sideeffect ".bork_directive /* ${0:c}:${1:c} */", "i,i,~{dirflag},~{fpsr},~{flags}"( i32 37, i32 927 )
        ret void

cond_false:
	call void asm sideeffect ".foo_directive ${0:c}:${1:c}", "i,i,~{dirflag},~{fpsr},~{flags}"( i32 37, i32 927 )
        call void asm sideeffect ".bork_directive /* ${0:c}:${1:c} */", "i,i,~{dirflag},~{fpsr},~{flags}"( i32 37, i32 927 )
        ret void
}
