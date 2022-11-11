; RUN: llc < %s -mtriple=i686-- -tailcallopt | FileCheck %s

declare i32 @putchar(i32)

define fastcc i32 @checktail(i32 %x, ptr %f, i32 %g) nounwind {
; CHECK-LABEL: checktail:
        %tmp1 = icmp sgt i32 %x, 0
        br i1 %tmp1, label %if-then, label %if-else

if-then:
        %arg1    = add i32 %x, -1
        call i32 @putchar(i32 90)       
; CHECK: jmpl *%e{{.*}}
        %res = tail call fastcc i32 %f( i32 %arg1, ptr %f, i32 %g)
        ret i32 %res

if-else:
        ret i32  %x
}


define i32 @main() nounwind { 
 %res = tail call fastcc i32 @checktail( i32 10, ptr @checktail,i32 10)
 ret i32 %res
}
