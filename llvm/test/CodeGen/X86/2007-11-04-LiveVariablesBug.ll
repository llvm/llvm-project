; RUN: llc -no-integrated-as < %s -mtriple=x86_64-unknown-linux-gnu
; PR1767

define void @xor_sse_2(i64 %bytes, ptr %p1, ptr %p2) {
entry:
        %p2_addr = alloca ptr          ; <ptr> [#uses=2]
        %lines = alloca i32             ; <ptr> [#uses=2]
        store ptr %p2, ptr %p2_addr, align 8
        %tmp1 = lshr i64 %bytes, 8              ; <i64> [#uses=1]
        %tmp12 = trunc i64 %tmp1 to i32         ; <i32> [#uses=2]
        store i32 %tmp12, ptr %lines, align 4
        %tmp6 = call ptr asm sideeffect "foo",
"=r,=*r,=*r,r,0,1,2,~{dirflag},~{fpsr},~{flags},~{memory}"(ptr elementtype(ptr) %p2_addr, ptr elementtype(i32) %lines, i64 256, ptr %p1, ptr %p2, i32 %tmp12 )              ; <ptr> [#uses=0]
        ret void
}
