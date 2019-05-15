; RUN: opt < %s -S -globalopt | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

@.str91250 = global [3 x i8] zeroinitializer

; CHECK: @A = local_unnamed_addr global i1 false
@A = global i1 icmp ne (i64 sub nsw (i64 ptrtoint (i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str91250, i64 0, i64 1) to i64), i64 ptrtoint ([3 x i8]* @.str91250 to i64)), i64 1)

; PR11352

@xs = global [2 x i32] zeroinitializer, align 4
; CHECK: @xs = local_unnamed_addr global [2 x i32] [i32 1, i32 1]

; PR12642
%PR12642.struct = type { i8 }
@PR12642.s = global <{}> zeroinitializer, align 1
@PR12642.p = constant %PR12642.struct* bitcast (i8* getelementptr (i8, i8* bitcast (<{}>* @PR12642.s to i8*), i64 1) to %PR12642.struct*), align 8

define internal void @test1() {
entry:
  store i32 1, i32* getelementptr inbounds ([2 x i32], [2 x i32]* @xs, i64 0, i64 0)
  %0 = load i32, i32* getelementptr inbounds ([2 x i32], [2 x i32]* @xs, i32 0, i64 0), align 4
  store i32 %0, i32* getelementptr inbounds ([2 x i32], [2 x i32]* @xs, i64 0, i64 1)
  ret void
}

; PR12060

%closure = type { i32 }

@f = internal global %closure zeroinitializer, align 4
@m = global i32 0, align 4
; CHECK-NOT: @f
; CHECK: @m = local_unnamed_addr global i32 13

define internal i32 @test2_helper(%closure* %this, i32 %b) {
entry:
  %0 = getelementptr inbounds %closure, %closure* %this, i32 0, i32 0
  %1 = load i32, i32* %0, align 4
  %add = add nsw i32 %1, %b
  ret i32 %add
}

define internal void @test2() {
entry:
  store i32 4, i32* getelementptr inbounds (%closure, %closure* @f, i32 0, i32 0)
  %call = call i32 @test2_helper(%closure* @f, i32 9)
  store i32 %call, i32* @m, align 4
  ret void
}

; PR19955

@dllimportptr = global i32* null, align 4
; CHECK: @dllimportptr = local_unnamed_addr global i32* null, align 4
@dllimportvar = external dllimport global i32
define internal void @test3() {
entry:
  store i32* @dllimportvar, i32** @dllimportptr, align 4
  ret void
}

@dllexportptr = global i32* null, align 4
; CHECK: @dllexportptr = local_unnamed_addr global i32* @dllexportvar, align 4
@dllexportvar = dllexport global i32 0, align 4
; CHECK: @dllexportvar = dllexport global i32 20, align 4
define internal void @test4() {
entry:
  store i32 20, i32* @dllexportvar, align 4
  store i32* @dllexportvar, i32** @dllexportptr, align 4
  ret void
}

@threadlocalptr = global i32* null, align 4
; CHECK: @threadlocalptr = global i32* null, align 4
@threadlocalvar = external thread_local global i32
define internal void @test5() {
entry:
  store i32* @threadlocalvar, i32** @threadlocalptr, align 4
  ret void
}

@test6_v1 = internal global { i32, i32 } { i32 42, i32 0 }, align 8
@test6_v2 = global i32 0, align 4
; CHECK: @test6_v2 = local_unnamed_addr global i32 42, align 4
define internal void @test6() {
  %load = load { i32, i32 }, { i32, i32 }* @test6_v1, align 8
  %xv0 = extractvalue { i32, i32 } %load, 0
  %iv = insertvalue { i32, i32 } %load, i32 %xv0, 1
  %xv1 = extractvalue { i32, i32 } %iv, 1
  store i32 %xv1, i32* @test6_v2, align 4
  ret void
}

@llvm.global_ctors = appending constant
  [6 x { i32, void ()*, i8* }]
  [{ i32, void ()*, i8* } { i32 65535, void ()* @test1, i8* null },
   { i32, void ()*, i8* } { i32 65535, void ()* @test2, i8* null },
   { i32, void ()*, i8* } { i32 65535, void ()* @test3, i8* null },
   { i32, void ()*, i8* } { i32 65535, void ()* @test4, i8* null },
   { i32, void ()*, i8* } { i32 65535, void ()* @test5, i8* null },
   { i32, void ()*, i8* } { i32 65535, void ()* @test6, i8* null }]
