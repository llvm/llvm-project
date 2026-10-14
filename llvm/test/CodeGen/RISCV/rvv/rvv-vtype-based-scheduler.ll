; RUN: llc -mtriple=riscv64 -mcpu=spacemit-x60 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefix=DEFAULT
; RUN: llc -mtriple=riscv64 -mcpu=spacemit-x60 -misched-prera-direction=bottomup \
; RUN:   -mattr=+enable-vsetvli-sched-heuristic -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefix=VTYPE-SCHED-BOTTOMUP
; RUN: llc -mtriple=riscv64 -mcpu=spacemit-x60 -misched-prera-direction=topdown \
; RUN:   -mattr=+enable-vsetvli-sched-heuristic -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefix=VTYPE-SCHED-TOPDOWN
; RUN: llc -mtriple=riscv64 -mcpu=spacemit-x60 -misched-prera-direction=bidirectional \
; RUN:   -mattr=+enable-vsetvli-sched-heuristic -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefix=VTYPE-SCHED-BIDIRECTIONAL

define void @test0(i16 %0, i16 %1, i16 %2, i16 %3, i16 %4, i16 %5, i16 %6, ptr %7, ptr %8, ptr %9, ptr %10, ptr %11, i32 %12) {
; DEFAULT-LABEL: test0:
; DEFAULT-COUNT-19: vset

; VTYPE-SCHED-BOTTOMUP-LABEL: test0:
; VTYPE-SCHED-BOTTOMUP-COUNT-15: vset

; VTYPE-SCHED-TOPDOWN-LABEL: test0:
; VTYPE-SCHED-TOPDOWN-COUNT-18: vset

; VTYPE-SCHED-BIDIRECTIONAL-LABEL: test0:
; VTYPE-SCHED-BIDIRECTIONAL-15: vset
entry:
  %14 = tail call <vscale x 8 x i8> @llvm.riscv.vle.nxv8i8.p0.i64(<vscale x 8 x i8> poison, ptr %7, i64 16)
  %15 = tail call <vscale x 8 x i8> @llvm.riscv.vle.nxv8i8.p0.i64(<vscale x 8 x i8> poison, ptr %8, i64 16)
  %16 = tail call <vscale x 8 x i8> @llvm.riscv.vle.nxv8i8.p0.i64(<vscale x 8 x i8> poison, ptr %9, i64 16)
  %17 = tail call <vscale x 8 x i8> @llvm.riscv.vle.nxv8i8.p0.i64(<vscale x 8 x i8> poison, ptr %10, i64 16)
  %18 = tail call <vscale x 4 x i16> @llvm.riscv.vmv.v.x.nxv4i16.i64(<vscale x 4 x i16> poison, i16 0, i64 8)
  %19 = trunc i16 %0 to i8
  %20 = tail call <vscale x 4 x i8> @llvm.vector.extract.nxv4i8.nxv8i8(<vscale x 8 x i8> %14, i64 0)
  %21 = tail call <vscale x 4 x i16> @llvm.riscv.vwmaccsu.nxv4i16.i8.nxv4i8.i64(<vscale x 4 x i16> %18, i8 %19, <vscale x 4 x i8> %20, i64 8, i64 3)
  %22 = tail call <vscale x 4 x i8> @llvm.vector.extract.nxv4i8.nxv8i8(<vscale x 8 x i8> %15, i64 0)
  %23 = tail call <vscale x 4 x i16> @llvm.riscv.vwmaccsu.nxv4i16.i8.nxv4i8.i64(<vscale x 4 x i16> %18, i8 %19, <vscale x 4 x i8> %22, i64 8, i64 3)
  %24 = tail call <vscale x 4 x i8> @llvm.vector.extract.nxv4i8.nxv8i8(<vscale x 8 x i8> %16, i64 0)
  %25 = tail call <vscale x 4 x i16> @llvm.riscv.vwmaccsu.nxv4i16.i8.nxv4i8.i64(<vscale x 4 x i16> %18, i8 %19, <vscale x 4 x i8> %24, i64 8, i64 3)
  %26 = tail call <vscale x 4 x i8> @llvm.vector.extract.nxv4i8.nxv8i8(<vscale x 8 x i8> %17, i64 0)
  %27 = tail call <vscale x 4 x i16> @llvm.riscv.vwmaccsu.nxv4i16.i8.nxv4i8.i64(<vscale x 4 x i16> %18, i8 %19, <vscale x 4 x i8> %26, i64 8, i64 3)
  %28 = tail call <vscale x 8 x i8> @llvm.riscv.vslidedown.nxv8i8.i64(<vscale x 8 x i8> poison, <vscale x 8 x i8> %14, i64 1, i64 8, i64 3)
  %29 = tail call <vscale x 8 x i8> @llvm.riscv.vslidedown.nxv8i8.i64(<vscale x 8 x i8> poison, <vscale x 8 x i8> %15, i64 1, i64 8, i64 3)
  %30 = tail call <vscale x 8 x i8> @llvm.riscv.vslidedown.nxv8i8.i64(<vscale x 8 x i8> poison, <vscale x 8 x i8> %16, i64 1, i64 8, i64 3)
  %31 = tail call <vscale x 8 x i8> @llvm.riscv.vslidedown.nxv8i8.i64(<vscale x 8 x i8> poison, <vscale x 8 x i8> %17, i64 1, i64 8, i64 3)
  %32 = trunc i16 %1 to i8
  %33 = tail call <vscale x 4 x i8> @llvm.vector.extract.nxv4i8.nxv8i8(<vscale x 8 x i8> %28, i64 0)
  %34 = tail call <vscale x 4 x i16> @llvm.riscv.vwmaccsu.nxv4i16.i8.nxv4i8.i64(<vscale x 4 x i16> %21, i8 %32, <vscale x 4 x i8> %33, i64 8, i64 3)
  %35 = tail call <vscale x 4 x i8> @llvm.vector.extract.nxv4i8.nxv8i8(<vscale x 8 x i8> %29, i64 0)
  %36 = tail call <vscale x 4 x i16> @llvm.riscv.vwmaccsu.nxv4i16.i8.nxv4i8.i64(<vscale x 4 x i16> %23, i8 %32, <vscale x 4 x i8> %35, i64 8, i64 3)
  %37 = tail call <vscale x 4 x i8> @llvm.vector.extract.nxv4i8.nxv8i8(<vscale x 8 x i8> %30, i64 0)
  %38 = tail call <vscale x 4 x i16> @llvm.riscv.vwmaccsu.nxv4i16.i8.nxv4i8.i64(<vscale x 4 x i16> %25, i8 %32, <vscale x 4 x i8> %37, i64 8, i64 3)
  %39 = tail call <vscale x 4 x i8> @llvm.vector.extract.nxv4i8.nxv8i8(<vscale x 8 x i8> %31, i64 0)
  %40 = tail call <vscale x 4 x i16> @llvm.riscv.vwmaccsu.nxv4i16.i8.nxv4i8.i64(<vscale x 4 x i16> %27, i8 %32, <vscale x 4 x i8> %39, i64 8, i64 3)
  %41 = tail call <vscale x 8 x i8> @llvm.riscv.vslidedown.nxv8i8.i64(<vscale x 8 x i8> poison, <vscale x 8 x i8> %14, i64 2, i64 8, i64 3)
  %42 = tail call <vscale x 8 x i8> @llvm.riscv.vslidedown.nxv8i8.i64(<vscale x 8 x i8> poison, <vscale x 8 x i8> %15, i64 2, i64 8, i64 3)
  %43 = tail call <vscale x 8 x i8> @llvm.riscv.vslidedown.nxv8i8.i64(<vscale x 8 x i8> poison, <vscale x 8 x i8> %16, i64 2, i64 8, i64 3)
  %44 = tail call <vscale x 8 x i8> @llvm.riscv.vslidedown.nxv8i8.i64(<vscale x 8 x i8> poison, <vscale x 8 x i8> %17, i64 2, i64 8, i64 3)
  %45 = trunc i16 %2 to i8
  %46 = tail call <vscale x 4 x i8> @llvm.vector.extract.nxv4i8.nxv8i8(<vscale x 8 x i8> %41, i64 0)
  %47 = tail call <vscale x 4 x i16> @llvm.riscv.vwmaccsu.nxv4i16.i8.nxv4i8.i64(<vscale x 4 x i16> %34, i8 %45, <vscale x 4 x i8> %46, i64 8, i64 3)
  %48 = tail call <vscale x 4 x i8> @llvm.vector.extract.nxv4i8.nxv8i8(<vscale x 8 x i8> %42, i64 0)
  %49 = tail call <vscale x 4 x i16> @llvm.riscv.vwmaccsu.nxv4i16.i8.nxv4i8.i64(<vscale x 4 x i16> %36, i8 %45, <vscale x 4 x i8> %48, i64 8, i64 3)
  %50 = tail call <vscale x 4 x i8> @llvm.vector.extract.nxv4i8.nxv8i8(<vscale x 8 x i8> %43, i64 0)
  %51 = tail call <vscale x 4 x i16> @llvm.riscv.vwmaccsu.nxv4i16.i8.nxv4i8.i64(<vscale x 4 x i16> %38, i8 %45, <vscale x 4 x i8> %50, i64 8, i64 3)
  %52 = tail call <vscale x 4 x i8> @llvm.vector.extract.nxv4i8.nxv8i8(<vscale x 8 x i8> %44, i64 0)
  %53 = tail call <vscale x 4 x i16> @llvm.riscv.vwmaccsu.nxv4i16.i8.nxv4i8.i64(<vscale x 4 x i16> %40, i8 %45, <vscale x 4 x i8> %52, i64 8, i64 3)
  %54 = tail call <vscale x 8 x i8> @llvm.riscv.vslidedown.nxv8i8.i64(<vscale x 8 x i8> poison, <vscale x 8 x i8> %14, i64 3, i64 8, i64 3)
  %55 = tail call <vscale x 8 x i8> @llvm.riscv.vslidedown.nxv8i8.i64(<vscale x 8 x i8> poison, <vscale x 8 x i8> %15, i64 3, i64 8, i64 3)
  %56 = tail call <vscale x 8 x i8> @llvm.riscv.vslidedown.nxv8i8.i64(<vscale x 8 x i8> poison, <vscale x 8 x i8> %16, i64 3, i64 8, i64 3)
  %57 = tail call <vscale x 8 x i8> @llvm.riscv.vslidedown.nxv8i8.i64(<vscale x 8 x i8> poison, <vscale x 8 x i8> %17, i64 3, i64 8, i64 3)
  %58 = trunc i16 %3 to i8
  %59 = tail call <vscale x 4 x i8> @llvm.vector.extract.nxv4i8.nxv8i8(<vscale x 8 x i8> %54, i64 0)
  %60 = tail call <vscale x 4 x i16> @llvm.riscv.vwmaccsu.nxv4i16.i8.nxv4i8.i64(<vscale x 4 x i16> %47, i8 %58, <vscale x 4 x i8> %59, i64 8, i64 3)
  %61 = tail call <vscale x 4 x i8> @llvm.vector.extract.nxv4i8.nxv8i8(<vscale x 8 x i8> %55, i64 0)
  %62 = tail call <vscale x 4 x i16> @llvm.riscv.vwmaccsu.nxv4i16.i8.nxv4i8.i64(<vscale x 4 x i16> %49, i8 %58, <vscale x 4 x i8> %61, i64 8, i64 3)
  %63 = tail call <vscale x 4 x i8> @llvm.vector.extract.nxv4i8.nxv8i8(<vscale x 8 x i8> %56, i64 0)
  %64 = tail call <vscale x 4 x i16> @llvm.riscv.vwmaccsu.nxv4i16.i8.nxv4i8.i64(<vscale x 4 x i16> %51, i8 %58, <vscale x 4 x i8> %63, i64 8, i64 3)
  %65 = tail call <vscale x 4 x i8> @llvm.vector.extract.nxv4i8.nxv8i8(<vscale x 8 x i8> %57, i64 0)
  %66 = tail call <vscale x 4 x i16> @llvm.riscv.vwmaccsu.nxv4i16.i8.nxv4i8.i64(<vscale x 4 x i16> %53, i8 %58, <vscale x 4 x i8> %65, i64 8, i64 3)
  %67 = tail call <vscale x 8 x i8> @llvm.riscv.vslidedown.nxv8i8.i64(<vscale x 8 x i8> poison, <vscale x 8 x i8> %14, i64 4, i64 8, i64 3)
  %68 = tail call <vscale x 8 x i8> @llvm.riscv.vslidedown.nxv8i8.i64(<vscale x 8 x i8> poison, <vscale x 8 x i8> %15, i64 4, i64 8, i64 3)
  %69 = tail call <vscale x 8 x i8> @llvm.riscv.vslidedown.nxv8i8.i64(<vscale x 8 x i8> poison, <vscale x 8 x i8> %16, i64 4, i64 8, i64 3)
  %70 = tail call <vscale x 8 x i8> @llvm.riscv.vslidedown.nxv8i8.i64(<vscale x 8 x i8> poison, <vscale x 8 x i8> %17, i64 4, i64 8, i64 3)
  %71 = trunc i16 %4 to i8
  %72 = tail call <vscale x 4 x i8> @llvm.vector.extract.nxv4i8.nxv8i8(<vscale x 8 x i8> %67, i64 0)
  %73 = tail call <vscale x 4 x i16> @llvm.riscv.vwmaccsu.nxv4i16.i8.nxv4i8.i64(<vscale x 4 x i16> %60, i8 %71, <vscale x 4 x i8> %72, i64 8, i64 3)
  %74 = tail call <vscale x 4 x i8> @llvm.vector.extract.nxv4i8.nxv8i8(<vscale x 8 x i8> %68, i64 0)
  %75 = tail call <vscale x 4 x i16> @llvm.riscv.vwmaccsu.nxv4i16.i8.nxv4i8.i64(<vscale x 4 x i16> %62, i8 %71, <vscale x 4 x i8> %74, i64 8, i64 3)
  %76 = tail call <vscale x 4 x i8> @llvm.vector.extract.nxv4i8.nxv8i8(<vscale x 8 x i8> %69, i64 0)
  %77 = tail call <vscale x 4 x i16> @llvm.riscv.vwmaccsu.nxv4i16.i8.nxv4i8.i64(<vscale x 4 x i16> %64, i8 %71, <vscale x 4 x i8> %76, i64 8, i64 3)
  %78 = tail call <vscale x 4 x i8> @llvm.vector.extract.nxv4i8.nxv8i8(<vscale x 8 x i8> %70, i64 0)
  %79 = tail call <vscale x 4 x i16> @llvm.riscv.vwmaccsu.nxv4i16.i8.nxv4i8.i64(<vscale x 4 x i16> %66, i8 %71, <vscale x 4 x i8> %78, i64 8, i64 3)
  %80 = tail call <vscale x 8 x i8> @llvm.riscv.vslidedown.nxv8i8.i64(<vscale x 8 x i8> poison, <vscale x 8 x i8> %14, i64 5, i64 8, i64 3)
  %81 = tail call <vscale x 8 x i8> @llvm.riscv.vslidedown.nxv8i8.i64(<vscale x 8 x i8> poison, <vscale x 8 x i8> %15, i64 5, i64 8, i64 3)
  %82 = tail call <vscale x 8 x i8> @llvm.riscv.vslidedown.nxv8i8.i64(<vscale x 8 x i8> poison, <vscale x 8 x i8> %16, i64 5, i64 8, i64 3)
  %83 = tail call <vscale x 8 x i8> @llvm.riscv.vslidedown.nxv8i8.i64(<vscale x 8 x i8> poison, <vscale x 8 x i8> %17, i64 5, i64 8, i64 3)
  %84 = trunc i16 %5 to i8
  %85 = tail call <vscale x 4 x i8> @llvm.vector.extract.nxv4i8.nxv8i8(<vscale x 8 x i8> %80, i64 0)
  %86 = tail call <vscale x 4 x i16> @llvm.riscv.vwmaccsu.nxv4i16.i8.nxv4i8.i64(<vscale x 4 x i16> %73, i8 %84, <vscale x 4 x i8> %85, i64 8, i64 3)
  %87 = tail call <vscale x 4 x i8> @llvm.vector.extract.nxv4i8.nxv8i8(<vscale x 8 x i8> %81, i64 0)
  %88 = tail call <vscale x 4 x i16> @llvm.riscv.vwmaccsu.nxv4i16.i8.nxv4i8.i64(<vscale x 4 x i16> %75, i8 %84, <vscale x 4 x i8> %87, i64 8, i64 3)
  %89 = tail call <vscale x 4 x i8> @llvm.vector.extract.nxv4i8.nxv8i8(<vscale x 8 x i8> %82, i64 0)
  %90 = tail call <vscale x 4 x i16> @llvm.riscv.vwmaccsu.nxv4i16.i8.nxv4i8.i64(<vscale x 4 x i16> %77, i8 %84, <vscale x 4 x i8> %89, i64 8, i64 3)
  %91 = tail call <vscale x 4 x i8> @llvm.vector.extract.nxv4i8.nxv8i8(<vscale x 8 x i8> %83, i64 0)
  %92 = tail call <vscale x 4 x i16> @llvm.riscv.vwmaccsu.nxv4i16.i8.nxv4i8.i64(<vscale x 4 x i16> %79, i8 %84, <vscale x 4 x i8> %91, i64 8, i64 3)
  %93 = tail call <vscale x 4 x i16> @llvm.riscv.vmax.nxv4i16.i16.i64(<vscale x 4 x i16> poison, <vscale x 4 x i16> %86, i16 0, i64 8)
  %94 = tail call <vscale x 4 x i16> @llvm.riscv.vmax.nxv4i16.i16.i64(<vscale x 4 x i16> poison, <vscale x 4 x i16> %88, i16 0, i64 8)
  %95 = tail call <vscale x 4 x i16> @llvm.riscv.vmax.nxv4i16.i16.i64(<vscale x 4 x i16> poison, <vscale x 4 x i16> %90, i16 0, i64 8)
  %96 = tail call <vscale x 4 x i16> @llvm.riscv.vmax.nxv4i16.i16.i64(<vscale x 4 x i16> poison, <vscale x 4 x i16> %92, i16 0, i64 8)
  %97 = tail call <vscale x 4 x i8> @llvm.riscv.vnclipu.nxv4i8.nxv4i16.i64.i64(<vscale x 4 x i8> poison, <vscale x 4 x i16> %93, i64 6, i64 0, i64 8)
  %98 = tail call <vscale x 4 x i8> @llvm.riscv.vnclipu.nxv4i8.nxv4i16.i64.i64(<vscale x 4 x i8> poison, <vscale x 4 x i16> %94, i64 6, i64 0, i64 8)
  %99 = tail call <vscale x 4 x i8> @llvm.riscv.vnclipu.nxv4i8.nxv4i16.i64.i64(<vscale x 4 x i8> poison, <vscale x 4 x i16> %95, i64 6, i64 0, i64 8)
  %100 = tail call <vscale x 4 x i8> @llvm.riscv.vnclipu.nxv4i8.nxv4i16.i64.i64(<vscale x 4 x i8> poison, <vscale x 4 x i16> %96, i64 6, i64 0, i64 8)
  tail call void @llvm.riscv.vse.nxv4i8.p0.i64(<vscale x 4 x i8> %97, ptr %11, i64 8)
  %101 = sext i32 %12 to i64
  %102 = getelementptr inbounds i8, ptr %11, i64 %101
  tail call void @llvm.riscv.vse.nxv4i8.p0.i64(<vscale x 4 x i8> %98, ptr %102, i64 8)
  %103 = shl nsw i32 %12, 1
  %104 = sext i32 %103 to i64
  %105 = getelementptr inbounds i8, ptr %11, i64 %104
  tail call void @llvm.riscv.vse.nxv4i8.p0.i64(<vscale x 4 x i8> %99, ptr %105, i64 8)
  %106 = mul nsw i32 %12, 3
  %107 = sext i32 %106 to i64
  %108 = getelementptr inbounds i8, ptr %11, i64 %107
  tail call void @llvm.riscv.vse.nxv4i8.p0.i64(<vscale x 4 x i8> %100, ptr %108, i64 8)
  ret void
}

define void @test1(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4) {
; DEFAULT-LABEL: test1:
; DEFAULT-COUNT-9: vset

; VTYPE-SCHED-BOTTOMUP-LABEL: test1:
; VTYPE-SCHED-BOTTOMUP-COUNT-5: vset

; VTYPE-SCHED-TOPDOWN-LABEL: test1:
; VTYPE-SCHED-TOPDOWN-COUNT-5: vset

; VTYPE-SCHED-BIDIRECTIONAL-LABEL: test1:
; VTYPE-SCHED-BIDIRECTIONAL-5: vset
entry:
  %5 = load <8 x i64>, ptr %1, align 64
  %6 = load <8 x i64>, ptr %2, align 64
  %7 = load <8 x i64>, ptr %3, align 64
  %8 = load <8 x i64>, ptr %4, align 64
  %9 = icmp ult <8 x i64> %5, %6
  %10 = bitcast <8 x i1> %9 to i8
  %11 = icmp eq <8 x i64> %5, %6
  %12 = bitcast <8 x i1> %11 to i8
  %13 = sub <8 x i64> %5, %6
  %14 = shl i8 %10, 1
  %15 = add i8 %14, %12
  %16 = xor i8 %15, %12
  %17 = bitcast i8 %16 to <8 x i1>
  %18 = sext <8 x i1> %17 to <8 x i64>
  %19 = add <8 x i64> %13, %18
  %20 = icmp ult <8 x i64> %7, %8
  %21 = bitcast <8 x i1> %20 to i8
  %22 = icmp eq <8 x i64> %7, %8
  %23 = bitcast <8 x i1> %22 to i8
  %24 = sub <8 x i64> %7, %8
  %25 = shl i8 %21, 1
  %26 = add i8 %25, %23
  %27 = xor i8 %26, %23
  %28 = bitcast i8 %27 to <8 x i1>
  %29 = sext <8 x i1> %28 to <8 x i64>
  %30 = add <8 x i64> %24, %29
  %31 = and <8 x i64> %30, %19
  store <8 x i64> %31, ptr %0, align 64
  ret void
}
