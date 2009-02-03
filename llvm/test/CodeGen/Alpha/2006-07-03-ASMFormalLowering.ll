; RUN: llvm-as < %s | llc -march=alpha

target datalayout = "e-p:64:64"
target triple = "alphaev67-unknown-linux-gnu"

define i32 @_ZN9__gnu_cxx18__exchange_and_addEPVii(i32* %__mem, i32 %__val) {
entry:
        %__tmp = alloca i32, align 4            ; <i32*> [#uses=1]
        %tmp3 = call i32 asm sideeffect "\0A$$Lxadd_0:\0A\09ldl_l  $0,$3\0A\09addl   $0,$4,$1\0A\09stl_c  $1,$2\0A\09beq    $1,$$Lxadd_0\0A\09mb", "=&r,=*&r,=*m,m,r"( i32* %__tmp, i32* %__mem, i32* %__mem, i32 %__val )            ; <i32> [#uses=1]
        ret i32 %tmp3
}

define void @_ZN9__gnu_cxx12__atomic_addEPVii(i32* %__mem, i32 %__val) {
entry:
        %tmp2 = call i32 asm sideeffect "\0A$$Ladd_1:\0A\09ldl_l  $0,$2\0A\09addl   $0,$3,$0\0A\09stl_c  $0,$1\0A\09beq    $0,$$Ladd_1\0A\09mb", "=&r,=*m,m,r"( i32* %__mem, i32* %__mem, i32 %__val )                ; <i32> [#uses=0]
        ret void
}

