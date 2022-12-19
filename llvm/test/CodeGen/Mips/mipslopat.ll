; This test does not check the machine code output.   
; RUN: llc -march=mips < %s 

@stat_vol_ptr_int = internal global ptr null, align 4
@stat_ptr_vol_int = internal global ptr null, align 4

define void @simple_vol_file() nounwind {
entry:
  %tmp = load volatile ptr, ptr @stat_vol_ptr_int, align 4
  call void @llvm.prefetch(ptr %tmp, i32 0, i32 0, i32 1)
  %tmp1 = load ptr, ptr @stat_ptr_vol_int, align 4
  call void @llvm.prefetch(ptr %tmp1, i32 0, i32 0, i32 1)
  ret void
}

declare void @llvm.prefetch(ptr nocapture, i32, i32, i32) nounwind

