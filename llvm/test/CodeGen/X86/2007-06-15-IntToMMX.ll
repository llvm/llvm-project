; RUN: llc < %s -march=x86-64 -mattr=+mmx | grep paddusw
@R = external global x86_mmx          ; <x86_mmx*> [#uses=1]

define void @foo(<1 x i64> %A, <1 x i64> %B) {
entry:
        %tmp2 = bitcast <1 x i64> %A to x86_mmx
        %tmp3 = bitcast <1 x i64> %B to x86_mmx
        %tmp7 = tail call x86_mmx @llvm.x86.mmx.paddus.w( x86_mmx %tmp2, x86_mmx %tmp3 )   ; <x86_mmx> [#uses=1]
        store x86_mmx %tmp7, x86_mmx* @R
        tail call void @llvm.x86.mmx.emms( )
        ret void
}

declare x86_mmx @llvm.x86.mmx.paddus.w(x86_mmx, x86_mmx)

declare void @llvm.x86.mmx.emms()
