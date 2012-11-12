; RUN: opt -std-compile-opts < %s | llvm-dis | not grep badref 

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.2"

%struct.anon = type { i32, i32 }
%struct.test = type { i64, %struct.anon, %struct.test* }

@TestArrayPtr = global %struct.test* getelementptr inbounds ([10 x %struct.test]* @TestArray, i64 0, i64 3) ; <%struct.test**> [#uses=1]
@TestArray = common global [10 x %struct.test] zeroinitializer, align 32 ; <[10 x %struct.test]*> [#uses=2]

define i32 @main() nounwind readonly {
  %diff1 = alloca i64                             ; <i64*> [#uses=2]
  call void @llvm.dbg.declare(metadata !{i64* %diff1}, metadata !0)
  store i64 72, i64* %diff1, align 8
  %v1 = load %struct.test** @TestArrayPtr, align 8 ; <%struct.test*> [#uses=1]
  %v2 = ptrtoint %struct.test* %v1 to i64 ; <i64> [#uses=1]
  %v3 = sub i64 %v2, ptrtoint ([10 x %struct.test]* @TestArray to i64) ; <i64> [#uses=1]
  store i64 %v3, i64* %diff1, align 8
  ret i32 4
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

!0 = metadata !{i32 459008, metadata !0, metadata !0, metadata !0, i32 38, metadata !0} ; [ DW_TAG_auto_variable ]
