; RUN: llc -mtriple=hexagon-unknown-linux-musl -mcpu=hexagonv68 \
; RUN:     -mattr=+hvxv68,+hvx-length128b -O2 -relocation-model=pic \
; RUN:     -frame-pointer=all -hexagon-small-data-threshold=0 \
; RUN:     -machine-sink-split=0 < %s -o /dev/null

; This testcase previously tripped the
; MachineRegisterInfo::updateDbgUsersToReg assertion
;   "Expected debug value to have some overlap with OldReg"
; during Machine Copy Propagation. The root cause was that W0-W15 had
; VF0-VF15 listed as an artificial (vsub_fake) sub-register, which
; caused MCP to associate DBG_VALUEs of $wN with unrelated $vM copies
; through reg-unit overlap.

define void @test(ptr %kernel, ptr %arrayidx, <32 x i32> %0) !dbg !3 {
entry:
  %1 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %0, <32 x i32> zeroinitializer, i32 0)
  %2 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> zeroinitializer, <32 x i32> %0, i32 0)
  %3 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> splat (i32 1), <32 x i32> zeroinitializer, i32 0)
  %4 = load <32 x i32>, ptr null, align 128
  %5 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %4, <32 x i32> zeroinitializer, i32 0)
  %6 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> zeroinitializer, <32 x i32> zeroinitializer, i32 0)
  %arrayidx77 = getelementptr i8, ptr %kernel, i32 3456
  %7 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> zeroinitializer, <32 x i32> zeroinitializer, i32 1)
  %e0 = shufflevector <64 x i32> %1, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %8 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> zeroinitializer, <32 x i32> %e0, i32 0)
  %e1 = shufflevector <64 x i32> %1, <64 x i32> zeroinitializer, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %9 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> zeroinitializer, <32 x i32> %e1, i32 0)
  %e2 = shufflevector <64 x i32> %2, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %e3 = shufflevector <64 x i32> %2, <64 x i32> zeroinitializer, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %10 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> zeroinitializer, <32 x i32> %e3, i32 0)
  %e4 = shufflevector <64 x i32> %3, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %e6 = shufflevector <64 x i32> %3, <64 x i32> zeroinitializer, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %11 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %e6, <32 x i32> zeroinitializer, i32 0)
  %e8 = shufflevector <64 x i32> %5, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %12 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> zeroinitializer, <32 x i32> %e8, i32 0)
  %e9 = shufflevector <64 x i32> %5, <64 x i32> zeroinitializer, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %13 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> zeroinitializer, <32 x i32> %e9, i32 0)
  %e12 = shufflevector <64 x i32> %6, <64 x i32> zeroinitializer, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %e14 = shufflevector <64 x i32> %6, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %14 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> zeroinitializer, <32 x i32> %e14, i32 0)
  %15 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> zeroinitializer, <32 x i32> %e12, i32 0)
  %e16 = shufflevector <64 x i32> %7, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %e18 = shufflevector <64 x i32> %7, <64 x i32> zeroinitializer, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %16 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %e18, <32 x i32> zeroinitializer, i32 0)
    #dbg_value(<64 x i32> %16, !16, !DIExpression(), !17)
  %17 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> zeroinitializer, <32 x i32> zeroinitializer, i32 -8)
  %e21 = shufflevector <64 x i32> %8, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %18 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %e2, <32 x i32> %e21, i32 0)
  %e23 = shufflevector <64 x i32> %8, <64 x i32> zeroinitializer, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %19 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %e2, <32 x i32> %e23, i32 0)
  %e24 = shufflevector <64 x i32> %10, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %e25 = shufflevector <64 x i32> %9, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %20 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %e24, <32 x i32> %e25, i32 0)
  %e26 = shufflevector <64 x i32> %10, <64 x i32> zeroinitializer, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %e27 = shufflevector <64 x i32> %9, <64 x i32> zeroinitializer, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %21 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %e26, <32 x i32> %e27, i32 0)
  %e28 = shufflevector <64 x i32> %12, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %22 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %e28, <32 x i32> %e4, i32 0)
  %e30 = shufflevector <64 x i32> %12, <64 x i32> zeroinitializer, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %23 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %e30, <32 x i32> %e4, i32 0)
  %e32 = shufflevector <64 x i32> %13, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %e33 = shufflevector <64 x i32> %11, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %24 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %e32, <32 x i32> %e33, i32 0)
  %e34 = shufflevector <64 x i32> %13, <64 x i32> zeroinitializer, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %e35 = shufflevector <64 x i32> %11, <64 x i32> zeroinitializer, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %25 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %e34, <32 x i32> %e35, i32 0)
  %e36 = shufflevector <64 x i32> %14, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %26 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %e36, <32 x i32> %e0, i32 0)
  %e38 = shufflevector <64 x i32> %14, <64 x i32> zeroinitializer, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %27 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %e38, <32 x i32> %e1, i32 0)
  %e42 = shufflevector <64 x i32> %15, <64 x i32> zeroinitializer, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %28 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %e42, <32 x i32> %e12, i32 0)
  %29 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> zeroinitializer, <32 x i32> %e16, i32 -16)
  %e46 = shufflevector <64 x i32> %17, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %e47 = shufflevector <64 x i32> %16, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %30 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %e46, <32 x i32> %e47, i32 0)
  %e48 = shufflevector <64 x i32> %17, <64 x i32> zeroinitializer, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %e49 = shufflevector <64 x i32> %16, <64 x i32> zeroinitializer, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %31 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %e48, <32 x i32> %e49, i32 0)
  %e50 = shufflevector <64 x i32> %22, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %e51 = shufflevector <64 x i32> %18, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %32 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %e50, <32 x i32> %e51, i32 0)
  %e52 = shufflevector <64 x i32> %23, <64 x i32> zeroinitializer, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %e53 = shufflevector <64 x i32> %19, <64 x i32> zeroinitializer, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %33 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %e52, <32 x i32> %e53, i32 0)
  %e54 = shufflevector <64 x i32> %24, <64 x i32> zeroinitializer, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %e55 = shufflevector <64 x i32> %20, <64 x i32> zeroinitializer, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %34 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %e54, <32 x i32> %e55, i32 0)
  %e56 = shufflevector <64 x i32> %25, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %e57 = shufflevector <64 x i32> %21, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %35 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %e56, <32 x i32> %e57, i32 0)
  %e59 = shufflevector <64 x i32> %29, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %e60 = shufflevector <64 x i32> %26, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %36 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %e59, <32 x i32> %e60, i32 0)
  %e62 = shufflevector <64 x i32> %27, <64 x i32> zeroinitializer, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %37 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %e18, <32 x i32> %e62, i32 0)
  %e63 = shufflevector <64 x i32> %30, <64 x i32> zeroinitializer, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %38 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %e63, <32 x i32> zeroinitializer, i32 0)
  %e65 = shufflevector <64 x i32> %31, <64 x i32> zeroinitializer, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %e66 = shufflevector <64 x i32> %28, <64 x i32> zeroinitializer, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %39 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %e65, <32 x i32> %e66, i32 0)
  %e67 = shufflevector <64 x i32> %36, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %e68 = shufflevector <64 x i32> %32, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %40 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %e67, <32 x i32> %e68, i32 0)
  %e69 = shufflevector <64 x i32> %37, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %e70 = shufflevector <64 x i32> %33, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %41 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %e69, <32 x i32> %e70, i32 0)
  %e71 = shufflevector <64 x i32> %38, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %e72 = shufflevector <64 x i32> %34, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %42 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %e71, <32 x i32> %e72, i32 0)
  %e74 = shufflevector <64 x i32> %39, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %43 = tail call <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32> %e74, <32 x i32> zeroinitializer, i32 0)
  %s0 = shufflevector <64 x i32> %40, <64 x i32> zeroinitializer, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  store <32 x i32> %s0, ptr %arrayidx, align 128
  %s1 = shufflevector <64 x i32> %41, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  store <32 x i32> %s1, ptr %kernel, align 128
  %s2 = shufflevector <64 x i32> %42, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  store <32 x i32> %s2, ptr %arrayidx, align 128
  %s3 = shufflevector <64 x i32> %35, <64 x i32> zeroinitializer, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  store <32 x i32> %s3, ptr %arrayidx77, align 128
  %s4 = shufflevector <64 x i32> %43, <64 x i32> zeroinitializer, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  store <32 x i32> %s4, ptr %kernel, align 128
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <64 x i32> @llvm.hexagon.V6.vshuffvdd.128B(<32 x i32>, <32 x i32>, i32)

; uselistorder directives
uselistorder ptr @llvm.hexagon.V6.vshuffvdd.128B, { 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "packetmath.cpp", directory: ".")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!4 = !DISubroutineType(types: !5)
!5 = !{null, !6}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 32)
!7 = !DICompositeType(tag: DW_TAG_array_type, baseType: !8, size: 2048, flags: DIFlagVector, elements: !9)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{!10}
!10 = !DISubrange(count: 64)
!11 = !{!12, !13, !14, !15, !16}
!12 = !DILocalVariable(name: "v0", scope: !3, file: !1, line: 2, type: !7)
!13 = !DILocalVariable(name: "v1", scope: !3, file: !1, line: 3, type: !7)
!14 = !DILocalVariable(name: "v2", scope: !3, file: !1, line: 4, type: !7)
!15 = !DILocalVariable(name: "v3", scope: !3, file: !1, line: 5, type: !7)
!16 = !DILocalVariable(name: "v4", scope: !3, file: !1, line: 6, type: !7)
!17 = !DILocation(line: 11, column: 1, scope: !3)
