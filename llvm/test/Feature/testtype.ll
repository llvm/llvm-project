; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

%X = type ptr addrspace(4)

        %inners = type { float, { i8 } }
        %struct = type { i32, %inners, i64 }

%fwd    = type ptr
%fwdref = type { ptr }

; same as above with unnamed types
%1 = type ptr 
%test = type %1
%0 = type { ptr }

%test2 = type [2 x i32]
;%x = type ptr

%test3 = type ptr
