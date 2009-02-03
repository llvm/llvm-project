; RUN: llvm-as < %s | llc -march=x86 -tailcallopt | grep {jmp} | grep {\\*%eax}

declare i32 @putchar(i32)

define fastcc i32 @checktail(i32 %x, i32* %f, i32 %g) {
        %tmp1 = icmp sgt i32 %x, 0
        br i1 %tmp1, label %if-then, label %if-else

if-then:
        %fun_ptr = bitcast i32* %f to i32(i32, i32*, i32)* 
        %arg1    = add i32 %x, -1
        call i32 @putchar(i32 90)       
        %res = tail call fastcc i32 %fun_ptr( i32 %arg1, i32 * %f, i32 %g)
        ret i32 %res

if-else:
        ret i32  %x
}


define i32 @main() { 
 %f   = bitcast i32 (i32, i32*, i32)* @checktail to i32*
 %res = tail call fastcc i32 @checktail( i32 10, i32* %f,i32 10)
 ret i32 %res
}