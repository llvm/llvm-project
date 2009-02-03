; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movups | count 2

define void @a(<4 x float>* %x) nounwind  {
entry:
        %tmp2 = load <4 x float>* %x, align 1
        %inv = call <4 x float> @llvm.x86.sse.rcp.ps(<4 x float> %tmp2)
        store <4 x float> %inv, <4 x float>* %x, align 1
        ret void
}

declare <4 x float> @llvm.x86.sse.rcp.ps(<4 x float>)
