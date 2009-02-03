; RUN: llvm-as < %s | llc -march=ia64

@_ZN9__gnu_cxx16__stl_prime_listE = external global [28 x i32]          ; <[28 x i32]*> [#uses=3]

define fastcc i32* @_ZSt11lower_boundIPKmmET_S2_S2_RKT0_(i32 %__val.val) {
entry:
        %retval = select i1 icmp slt (i32 ashr (i32 sub (i32 ptrtoint (i32* getelementptr ([28 x i32]* @_ZN9__gnu_cxx16__stl_prime_listE, i32 0, i32 28) to i32), i32 ptrtoint ([28 x i32]* @_ZN9__gnu_cxx16__stl_prime_listE to i32)), i32 2), i32 0), i32* null, i32* getelementptr ([28 x i32]* @_ZN9__gnu_cxx16__stl_prime_listE, i32 0, i32 0)         ; <i32*> [#uses=1]
        ret i32* %retval
}

