; RUN: llvm-as < %s | llc
	
define void @""(float* %inregs, float* %outregs) {
        %a_addr.i = alloca <4 x float>          ; <<4 x float>*> [#uses=1]
        store <4 x float> < float undef, float undef, float undef, float undef >, <4 x float>* %a_addr.i
        ret void
}
