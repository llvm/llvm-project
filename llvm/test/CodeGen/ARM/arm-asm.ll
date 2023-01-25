; RUN: llc -mtriple=arm-eabi %s -o /dev/null

define void @frame_dummy() {
entry:
        %tmp1 = tail call ptr (ptr) asm "", "=r,0,~{dirflag},~{fpsr},~{flags}"( ptr null )           ; <ptr> [#uses=0]
        ret void
}
