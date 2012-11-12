; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu
; PR1767

define void @xor_sse_2(i64 %bytes, i64* %p1, i64* %p2) {
entry:
        %p2_addr = alloca i64*          ; <i64**> [#uses=2]
        %lines = alloca i32             ; <i32*> [#uses=2]
        store i64* %p2, i64** %p2_addr, align 8
        %tmp1 = lshr i64 %bytes, 8              ; <i64> [#uses=1]
        %tmp12 = trunc i64 %tmp1 to i32         ; <i32> [#uses=2]
        store i32 %tmp12, i32* %lines, align 4
        %tmp6 = call i64* asm sideeffect "foo",
"=r,=*r,=*r,r,0,1,2,~{dirflag},~{fpsr},~{flags},~{memory}"( i64** %p2_addr,
i32* %lines, i64 256, i64* %p1, i64* %p2, i32 %tmp12 )              ; <i64*> [#uses=0]
        ret void
}
