; RUN: opt -passes="always-inline" -pass-remarks=inline -pass-remarks-missed=inline -S < %s 2>&1 | FileCheck %s --implicit-check-not="remark: "

declare void @personalityFn1();
declare void @personalityFn2();

define void @foo() alwaysinline {
    ret void
}

define void @bar() alwaysinline personality ptr @personalityFn1 {
    ret void
}

define void @goo() personality ptr @personalityFn2 {
    ; CHECK-DAG: remark: {{.*}}: 'bar' is not inlined into 'goo': incompatible personality
    call void @bar()
    ; CHECK-DAG: remark: {{.*}}: 'foo' is not inlined into 'goo': unsupported operand bundle
    call void @foo() [ "CUSTOM_OPERAND_BUNDLE"() ]
    ; CHECK-DAG: remark: {{.*}}: 'foo' inlined into 'goo' with (cost=always): always inline attribute
    call void @foo()
    ret void
}
