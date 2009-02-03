; RUN: llvm-as < %s | opt -verify | llvm-dis | llvm-as -disable-output

define {i32, i8} @foo(i32 %p) {
  ret i32 1, i8 2
}

define i8 @f2(i32 %p) {
   %c = call {i32, i8} @foo(i32 %p)
   %d = getresult {i32, i8} %c, 1
   %e = add i8 %d, 1
   ret i8 %e
}

define i32 @f3(i32 %p) {
   %c = invoke {i32, i8} @foo(i32 %p)
         to label %L unwind label %L2
   L: 
   %d = getresult {i32, i8} %c, 0
   ret i32 %d
   L2:
   ret i32 0
}
