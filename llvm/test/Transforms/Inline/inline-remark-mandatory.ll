; RUN: opt -passes="cgscc(inline<only-mandatory>)" -pass-remarks-missed=inline -S < %s 2>&1 | FileCheck %s

declare void @personalityFn1();
declare void @personalityFn2();

define i32 @a() personality void ()* @personalityFn1 {
    ret i32 1
}

define i32 @b() personality void ()* @personalityFn2 {
    %r = call i32 @a() alwaysinline
    ret i32 %r
}

; CHECK: remark: {{.*}} 'a' is not AlwaysInline into 'b': incompatible personality

