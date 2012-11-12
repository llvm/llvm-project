; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

@foo1 = external global <4 x float>
@foo2 = external global <2 x i10>


define void @main() 
{
        store <4 x float> <float 1.0, float 2.0, float 3.0, float 4.0>, <4 x float>* @foo1
        store <2 x i10> <i10 4, i10 4>, <2 x i10>* @foo2
	%l1 = load <4 x float>* @foo1
        %l2 = load <2 x i10>* @foo2
        ret void
}
