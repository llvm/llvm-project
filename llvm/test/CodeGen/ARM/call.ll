; RUN: llvm-as < %s | llc -march=arm | grep {mov lr, pc}
; RUN: llvm-as < %s | llc -march=arm -mattr=+v5t | grep blx
; RUN: llvm-as < %s | llc -march=arm -mtriple=arm-linux-gnueabi\
; RUN:   -relocation-model=pic | grep {PLT}

@t = weak global i32 ()* null           ; <i32 ()**> [#uses=1]

declare void @g(i32, i32, i32, i32)

define void @f() {
        call void @g( i32 1, i32 2, i32 3, i32 4 )
        ret void
}

define void @g.upgrd.1() {
        %tmp = load i32 ()** @t         ; <i32 ()*> [#uses=1]
        %tmp.upgrd.2 = tail call i32 %tmp( )            ; <i32> [#uses=0]
        ret void
}
