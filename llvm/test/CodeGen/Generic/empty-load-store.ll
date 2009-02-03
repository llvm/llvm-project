; RUN: llvm-as < %s | llc
; PR2612

@current_foo = internal global {  } zeroinitializer

define i32 @foo() {
entry:
        %retval = alloca i32
        store i32 0, i32* %retval
        %local_foo = alloca {  }
        load {  }* @current_foo
        store {  } %0, {  }* %local_foo
        br label %return

return:
        load i32* %retval
        ret i32 %1
}
