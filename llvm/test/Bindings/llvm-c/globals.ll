; RUN: llvm-as < %s | llvm-c-test --module-list-globals | FileCheck %s

@foo = constant [7 x i8] c"foobar\00", align 1
;CHECK: GlobalDefinition: foo ptr

@bar = common global i32 0, align 4
;CHECK: GlobalDefinition: bar ptr
