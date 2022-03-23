; RUN: llvm-as -disable-output %s 2>&1
; FIXME: this should err out saying not to mix `ptr` and `foo*`
define void @f(ptr) {
	%a = alloca i32*
	ret void
}
