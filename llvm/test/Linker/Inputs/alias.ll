@zed = global i32 42
@foo = alias i32, ptr @zed
@foo2 = alias i16, ptr @zed
