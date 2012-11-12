; RUN: llc < %s -O3 -mtriple=thumbv7-apple-darwin10 -mcpu=cortex-a8 -relocation-model=pic
; PR7484

%struct.gs_matrix = type { float, i32, float, i32, float, i32, float, i32, float, i32, float, i32 }

define fastcc void @func(%struct.gs_matrix* nocapture %pm1) nounwind {
entry:
  %0 = getelementptr inbounds %struct.gs_matrix* %pm1, i32 0, i32 6
  %1 = load float* %0, align 4
  %2 = getelementptr inbounds %struct.gs_matrix* %pm1, i32 0, i32 8
  %3 = load float* %2, align 4
  %4 = getelementptr inbounds %struct.gs_matrix* %pm1, i32 0, i32 2
  %5 = bitcast float* %4 to i32*
  %6 = load i32* %5, align 4
  %7 = or i32 0, %6
  %.mask = and i32 %7, 2147483647
  %8 = icmp eq i32 %.mask, 0
  br i1 %8, label %bb, label %bb11

bb:
  ret void

bb11:
  %9 = fmul float %1, undef
  %10 = fmul float %3, undef
  ret void
}
