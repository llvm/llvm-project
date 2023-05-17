; RUN: llc < %s -O3 -mtriple=thumbv7-apple-darwin10 -mcpu=cortex-a8 -relocation-model=pic
; PR7484

%struct.gs_matrix = type { float, i32, float, i32, float, i32, float, i32, float, i32, float, i32 }

define fastcc void @func(ptr nocapture %pm1) nounwind {
entry:
  %0 = getelementptr inbounds %struct.gs_matrix, ptr %pm1, i32 0, i32 6
  %1 = load float, ptr %0, align 4
  %2 = getelementptr inbounds %struct.gs_matrix, ptr %pm1, i32 0, i32 8
  %3 = load float, ptr %2, align 4
  %4 = getelementptr inbounds %struct.gs_matrix, ptr %pm1, i32 0, i32 2
  %5 = load i32, ptr %4, align 4
  %6 = or i32 0, %5
  %.mask = and i32 %6, 2147483647
  %7 = icmp eq i32 %.mask, 0
  br i1 %7, label %bb, label %bb11

bb:
  ret void

bb11:
  %8 = fmul float %1, undef
  %9 = fmul float %3, undef
  ret void
}
