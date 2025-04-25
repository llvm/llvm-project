; RUN: opt -S -passes='dxil-legalize' -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

define void @const_i8_store() {
    %accum.i.flat = alloca [1 x i32], align 4
    %i = alloca i8, align 4
    store i8 1, ptr %i
    %i8.load = load i8, ptr %i
    %z = zext i8 %i8.load to i32
    %gep = getelementptr i32, ptr %accum.i.flat, i32 0
    store i32 %z, ptr %gep, align 4
    ret void
}

define void @const_add_i8_store() {
    %accum.i.flat = alloca [1 x i32], align 4
    %i = alloca i8, align 4
    %add_i8 = add nsw i8 3, 1
    store i8 %add_i8, ptr %i
    %i8.load = load i8, ptr %i
    %z = zext i8 %i8.load to i32
    %gep = getelementptr i32, ptr %accum.i.flat, i32 0
    store i32 %z, ptr %gep, align 4
    ret void
}

define void @var_i8_store(i1 %cmp.i8) {
    %accum.i.flat = alloca [1 x i32], align 4
    %i = alloca i8, align 4
    %select.i8 = select i1 %cmp.i8, i8 1, i8 2
    store i8 %select.i8, ptr %i
    %i8.load = load i8, ptr %i
    %z = zext i8 %i8.load to i32
    %gep = getelementptr i32, ptr %accum.i.flat, i32 0
    store i32 %z, ptr %gep, align 4
    ret void
}

define void @conflicting_cast(i1 %cmp.i8) {
    %accum.i.flat = alloca [2 x i32], align 4
    %i = alloca i8, align 4
    %select.i8 = select i1 %cmp.i8, i8 1, i8 2
    store i8 %select.i8, ptr %i
    %i8.load = load i8, ptr %i
    %z = zext i8 %i8.load to i16
    %gep1 = getelementptr i16, ptr %accum.i.flat, i32 0
    store i16 %z, ptr %gep1, align 2
    %gep2 = getelementptr i16, ptr %accum.i.flat, i32 1
    store i16 %z, ptr %gep2, align 2
    %z2 = zext i8 %i8.load to i32
    %gep3 = getelementptr i32, ptr %accum.i.flat, i32 1
    store i32 %z2, ptr %gep3, align 4
    ret void
}