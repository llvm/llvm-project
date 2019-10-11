; RUN: llc < %s -mtriple=i686-- | FileCheck %s

declare i32 @putchar(i32)

define tailcc i32 @checktail(i32 %x, i32* %f, i32 %g) nounwind {
; CHECK-LABEL: checktail:
        %tmp1 = icmp sgt i32 %x, 0
        br i1 %tmp1, label %if-then, label %if-else

if-then:
        %fun_ptr = bitcast i32* %f to i32(i32, i32*, i32)* 
        %arg1    = add i32 %x, -1
        call i32 @putchar(i32 90)       
; CHECK: jmpl *%e{{.*}}
        %res = tail call tailcc i32 %fun_ptr( i32 %arg1, i32 * %f, i32 %g)
        ret i32 %res

if-else:
        ret i32  %x
}


define i32 @main() nounwind { 
 %f   = bitcast i32 (i32, i32*, i32)* @checktail to i32*
 %res = tail call tailcc i32 @checktail( i32 10, i32* %f,i32 10)
 ret i32 %res
}
