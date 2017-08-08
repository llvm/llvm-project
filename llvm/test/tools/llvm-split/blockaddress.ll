; Test that blockaddress target is in the same partition.
; RUN: llvm-split -j5 -o %t %s
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1234 %s
; RUN: llvm-dis -o - %t2 | FileCheck --check-prefix=CHECK1234 %s
; RUN: llvm-dis -o - %t3 | FileCheck --check-prefix=CHECK1234 %s
; RUN: llvm-dis -o - %t4 | FileCheck --check-prefix=CHECK1234 %s

; CHECK0:    @xxx = global [2 x i8*] [i8* blockaddress(@f, %exit), i8* blockaddress(@g, %exit)]
; CHECK1234: @xxx = external global [2 x i8*]
; CHECK1234-NOT: blockaddress
@xxx = global [2 x i8*] [i8* blockaddress(@f, %exit), i8* blockaddress(@g, %exit)]

; CHECK0:    define i32 @f()
; CHECK1234: declare i32 @f()
define i32 @f(){
entry:
  br label %exit
exit:
  ret i32 0
}

; CHECK0:    define i32 @g()
; CHECK1234: declare i32 @g()
define i32 @g(){
entry:
  br label %exit
exit:
  ret i32 0
}

; CHECK0:    define i8* @h()
; CHECK1234: declare i8* @h()
define i8* @h(){
entry:
  ret i8* blockaddress(@f, %exit)
}
