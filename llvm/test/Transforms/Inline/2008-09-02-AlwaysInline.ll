; RUN: llvm-as < %s | opt  -inline-threshold=0 -inline | llvm-dis | not grep call 

define i32 @fn2() alwaysinline {
  ret i32 1
}

define i32 @fn3() {
   %r = call i32 @fn2()
   ret i32 %r
}
