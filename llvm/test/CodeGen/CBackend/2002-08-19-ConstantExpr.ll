; RUN: llvm-as < %s | llc -march=c

global i32* bitcast (float* @2 to i32*)   ;; Forward numeric reference
global float* @2                       ;; Duplicate forward numeric reference
global float 0.0

@array = constant [2 x i32] [ i32 12, i32 52 ]
@arrayPtr = global i32* getelementptr ([2 x i32]* @array, i64 0, i64 0)
