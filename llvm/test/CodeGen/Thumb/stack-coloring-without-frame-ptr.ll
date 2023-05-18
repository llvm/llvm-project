; RUN: llc -mtriple=thumb-eabi -mcpu=arm1022e %s -o /dev/null

%iterator = type { ptr, ptr, ptr, ptr }
%insert_iterator = type { ptr, %iterator }
%deque = type { %iterator, %iterator, ptr, i32 }

define i32 @test_thumbv5e_fp_elim() nounwind optsize {
entry:
  %var1 = alloca %iterator, align 4
  %var2 = alloca %insert_iterator, align 4
  %var3 = alloca %deque, align 4

  call void @llvm.lifetime.start.p0(i64 16, ptr %var1) nounwind
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %var1, ptr align 4 %var3, i32 16, i1 false)
  call void @llvm.lifetime.end.p0(i64 16, ptr %var1) nounwind

  call void @llvm.lifetime.start.p0(i64 20, ptr %var2) nounwind

  ret i32 0
}

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1) nounwind

declare void @llvm.lifetime.start.p0(i64, ptr nocapture) nounwind

declare void @llvm.lifetime.end.p0(i64, ptr nocapture) nounwind
