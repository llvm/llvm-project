; RUN: opt -S -passes=globalopt < %s | FileCheck %s

target datalayout = "p:32:32"

%struct.s.2 = type { %struct.t.1, %struct.t.1, %struct.t.1, %struct.u.0, %struct.u.0 }
%struct.t.1 = type { %struct.u.0, %struct.u.0, %struct.u.0, %struct.u.0, i32, i32, i32, i32 }
%struct.u.0 = type { i32, i32, i32, i8 }

@s = external global [700 x [24000 x %struct.s.2]], align 1
@p = global %struct.s.2* bitcast (i8* getelementptr (i8, i8* bitcast ([700 x [24000 x %struct.s.2]]* @s to i8*), i64 2247483647) to %struct.s.2*), align 1

; CHECK: @p = local_unnamed_addr global %struct.s.2* bitcast (i8* getelementptr (i8, i8* bitcast ([700 x [24000 x %struct.s.2]]* @s to i8*), i32 -2047483649) to %struct.s.2*), align 1
