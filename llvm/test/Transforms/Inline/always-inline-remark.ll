; RUN: opt -passes="always-inline" -pass-remarks-missed=inline -S < %s 2>&1 | FileCheck %s

declare void @personalityFn1();
declare void @personalityFn2();

define void @foo() alwaysinline {
    ret void
}

define void @bar() alwaysinline personality void ()* @personalityFn1 {
    ret void
}

define void @goo() personality void ()* @personalityFn2 {
    ; CHECK-DAG: 'bar' is not inlined into 'goo': incompatible personality
    call void @bar()
    ; CHECK-DAG: 'foo' is not inlined into 'goo': unsupported operand bundle
    call void @foo() [ "CUSTOM_OPERAND_BUNDLE"() ]
    ret void
}
