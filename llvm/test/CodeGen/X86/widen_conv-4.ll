; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse42 -disable-mmx -o %t -f

; unsigned to float v7i16 to v7f32

define void @convert(<7 x float>* %dst.addr, <7 x i16> %src) nounwind {
entry:
	%val = sitofp <7 x i16> %src to <7 x float>
	store <7 x float> %val, <7 x float>* %dst.addr
	ret void
}
