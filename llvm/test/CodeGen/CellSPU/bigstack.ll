; RUN: llc < %s -march=cellspu -o %t1.s
; RUN: grep lqx   %t1.s | count 3
; RUN: grep il    %t1.s | grep -v file | count 5
; RUN: grep stqx  %t1.s | count 1

define i32 @bigstack() nounwind {
entry:
  %avar = alloca i32                            
  %big_data = alloca [2048 x i32]                
  store i32 3840, i32* %avar, align 4
  br label %return

return:                                          
  %retval = load i32* %avar                
  ret i32 %retval
}

