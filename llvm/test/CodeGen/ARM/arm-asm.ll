; RUN: llc < %s -march=arm

define void @frame_dummy() {
entry:
        %tmp1 = tail call void (i8*)* (void (i8*)*)* asm "", "=r,0,~{dirflag},~{fpsr},~{flags}"( void (i8*)* null )           ; <void (i8*)*> [#uses=0]
        ret void
}
