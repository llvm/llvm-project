; RUN: llc < %s
; PR2612

; Triggers a crash on assertion as NVPTX does not support 0-sized arrays.
; UNSUPPORTED: target=nvptx{{.*}}

@current_foo = internal global {  } zeroinitializer

define i32 @foo() {
entry:
        %retval = alloca i32
        store i32 0, ptr %retval
        %local_foo = alloca {  }
        load {  }, ptr @current_foo
        store {  } %0, ptr %local_foo
        br label %return

return:
        load i32, ptr %retval
        ret i32 %1
}
