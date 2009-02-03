; RUN: llvm-as < %s > %t.bc
; RUN: llvm-ld %t.bc -o %t.sh
; PR2612

@current_foo = internal global {  } zeroinitializer

define i32 @main(...) {
entry:
        %retval = alloca i32            ; <i32*> [#uses=2]
        store i32 0, i32* %retval
        %local_foo = alloca {  }                ; <{  }*> [#uses=1]
        load {  }* @current_foo         ; <{  }>:0 [#uses=1]
        store {  } %0, {  }* %local_foo
        br label %return

return:         ; preds = %entry
        load i32* %retval               ; <i32>:1 [#uses=1]
        ret i32 %1
}

