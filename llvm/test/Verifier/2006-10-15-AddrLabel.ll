; RUN: not llvm-as < %s > /dev/null |& grep {basic block pointers are invalid}

define i32 @main() {
         %foo  = call i8* %llvm.stacksave()
         %foop = bitcast i8* %foo to label*
         %nret = load label* %foop
         br label %nret;
}
