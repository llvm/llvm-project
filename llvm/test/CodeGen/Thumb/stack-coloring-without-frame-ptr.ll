; RUN: llc -mtriple=thumb-eabi -mcpu=arm1022e %s -o /dev/null

%iterator = type { i8**, i8**, i8**, i8*** }
%insert_iterator = type { %deque*, %iterator }
%deque = type { %iterator, %iterator, i8***, i32 }

define i32 @test_thumbv5e_fp_elim() nounwind optsize {
entry:
  %var1 = alloca %iterator, align 4
  %var2 = alloca %insert_iterator, align 4
  %var3 = alloca %deque, align 4

  %0 = bitcast %deque* %var3 to i8*
  %1 = bitcast %iterator* %var1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %1) nounwind
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %1, i8* align 4 %0, i32 16, i1 false)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %1) nounwind

  %2 = bitcast %insert_iterator* %var2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 20, i8* %2) nounwind

  ret i32 0
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) nounwind

declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) nounwind
