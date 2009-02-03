; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

%X = type i32* addrspace(4)*

        %inners = type { float, { i8 } }
        %struct = type { i32, %inners, i64 }

%fwdref = type { %fwd* }
%fwd    = type %fwdref*

; same as above with unnamed types
type { %1* }
type %0* 
%test = type %1

%test2 = type [2 x i32]
;%x = type %undefined*

%test3 = type i32 (i32()*, float(...)*, ...)*
