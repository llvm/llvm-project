
@foo = alias i32 (...), @foo2

define linkonce_odr i32 @foo2() {
    %ret = add i32 42, 42
    ret i32 %ret
}