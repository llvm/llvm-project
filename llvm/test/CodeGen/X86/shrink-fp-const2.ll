; RUN: llvm-as < %s | llc -march=x86 | grep flds
; This should be a flds, not fldt.
define x86_fp80 @test2() nounwind  {
entry:
	ret x86_fp80 0xK3FFFC000000000000000
}

