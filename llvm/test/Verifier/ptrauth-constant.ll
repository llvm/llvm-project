; RUN: not opt -passes=verify < %s 2>&1 | FileCheck %s

@g = external global i8

; CHECK: signed ptrauth constant deactivation symbol must be a global value or null
@ptr = global ptr ptrauth (ptr @g, i32 0, i64 65535, ptr null, ptr inttoptr (i64 16 to ptr))
