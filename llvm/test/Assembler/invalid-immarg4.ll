; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: Attribute 'range(i32 1, 145)' applied to incompatible type!
declare void @llvm.test.immarg.range.intrinsic.f32(float immarg range(i32 1, 145))
