; RUN: llvm-as < %s | llc
; PR1845

define void @boolVectorSelect(<4 x i1>* %boolVectorPtr) {
Body:
        %castPtr = bitcast <4 x i1>* %boolVectorPtr to <4 x i1>*
        %someBools = load <4 x i1>* %castPtr, align 1           ; <<4 x i1>>
        %internal = alloca <4 x i1>, align 16           ; <<4 x i1>*> [#uses=1]
        store <4 x i1> %someBools, <4 x i1>* %internal, align 1
        ret void
}
