; RUN: llc < %s  -mtriple=i686-unknown-linux  -tailcallopt | FileCheck %s
; Linux has 8 byte alignment so the params cause stack size 20 when tailcallopt
; is enabled, ensure that a normal fastcc call has matching stack size


define fastcc i32 @tailcallee(i32 %a1, i32 %a2, i32 %a3, i32 %a4) {
       ret i32 %a3
}

define fastcc i32 @tailcaller(i32 %in1, i32 %in2, i32 %in3, i32 %in4) {
       %tmp11 = tail call fastcc i32 @tailcallee(i32 %in1, i32 %in2,
                                                 i32 %in1, i32 %in2)
       ret i32 %tmp11
}

define i32 @main(i32 %argc, i8** %argv) {
 %tmp1 = call fastcc i32 @tailcaller( i32 1, i32 2, i32 3, i32 4 )
 ; expect match subl [stacksize] here
 ret i32 0
}

; CHECK: calll tailcaller
; CHECK-NEXT: subl $12
