; This test does not check the machine code output.   
; RUN: llc -march=mips < %s 

@stat_vol_ptr_int = internal global i32* null, align 4
@stat_ptr_vol_int = internal global i32* null, align 4

define void @simple_vol_file() nounwind {
entry:
  %tmp = load volatile i32** @stat_vol_ptr_int, align 4
  %0 = bitcast i32* %tmp to i8*
  call void @llvm.prefetch(i8* %0, i32 0, i32 0, i32 1)
  %tmp1 = load i32** @stat_ptr_vol_int, align 4
  %1 = bitcast i32* %tmp1 to i8*
  call void @llvm.prefetch(i8* %1, i32 0, i32 0, i32 1)
  ret void
}

declare void @llvm.prefetch(i8* nocapture, i32, i32, i32) nounwind

