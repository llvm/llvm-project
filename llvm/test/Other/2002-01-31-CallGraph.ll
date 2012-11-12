;  Call graph construction crash: Not handling indirect calls right
;
; RUN: opt < %s -analyze -print-callgraph >& /dev/null
;

        %FunTy = type i32 (i32)

define void @invoke(%FunTy* %x) {
        %foo = call i32 %x( i32 123 )           ; <i32> [#uses=0]
        ret void
}


