; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse42 -disable-mmx -o %t -f
; grep cvtsi2ss  %t | count 1 
; sign to float v2i16 to v2f32

define void @convert(<2 x float>* %dst.addr, <2 x i16> %src) nounwind {
entry:
	%val = sitofp <2 x i16> %src to <2 x float>
	store <2 x float> %val, <2 x float>* %dst.addr
	ret void
}
