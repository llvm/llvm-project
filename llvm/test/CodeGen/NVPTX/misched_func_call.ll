; RUN: llc -O3 -march=nvptx64 -enable-misched %s -o - | FileCheck %s

; ModuleID = 'The Accel Module'
source_filename = "The Accel Module"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: noinline
define ptx_kernel void @"my_kernel"(i32 %"arg_0", i64 %"arg_1", i64 %"arg_2", i64 %"arg_3") {
"Entry_BB":
%r = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
%r6 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
%r7 = mul i32 %r, %r6
%r9 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
%r10 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
%r11 = mul i32 %r, %r9
%r12 = add i32 %r10, %r11
%"arg_3.tr" = trunc i64 %"arg_3" to i32
%r16 = shl i32 %"arg_3.tr", 1
%r19.not = icmp slt i32 %r12, %r16
br i1 %r19.not, label %"BB1490", label %"EXIT_BB"

"BB1490": ; preds = %"Entry_BB"
%r23 = sext i32 %"arg_0" to i64
%r24 = shl nsw i64 %r23, 3
br label %"BB1692"

"BB1692": ; preds = %"BB18", %"BB1490"
%"$$i_l40_0_t23.0" = phi i32 [ %r12, %"BB1490" ], [ %r80, %"BB18" ]
%r28 = sext i32 %"$$i_l40_0_t23.0" to i64
%0 = or i64 %r28, %"arg_3"
%1 = and i64 %0, -4294967296
%2 = icmp eq i64 %1, 0
br i1 %2, label %3, label %8

3:                                                ; preds = %"BB1692"
%4 = trunc i64 %"arg_3" to i32
%5 = trunc i64 %r28 to i32
%6 = udiv i32 %5, %4
%7 = zext i32 %6 to i64
br label %"BB18"

8:                                                ; preds = %"BB1692"
%9 = sdiv i64 %r28, %"arg_3"
br label %"BB18"

"BB18": ; preds = %8, %3
%10 = phi i64 [ %7, %3 ], [ %9, %8 ]
%r31 = trunc i64 %10 to i32
%.neg = mul i64 %10, -4294967296
%r35 = ashr exact i64 %.neg, 32
%r38 = mul i64 %"arg_3", %r35
%r39 = add i64 %r28, %r38
%r42 = mul i64 %r24, %r39
%r44 = mul i32 %r31, 10
%r47 = inttoptr i64 %"arg_1" to ptr addrspace(1)
%gep2 = getelementptr i8, ptr addrspace(1) %r47, i64 %r42
%11 = sext i32 %r44 to i64
%r53 = getelementptr double, ptr addrspace(1) %gep2, i64 %11
%r54 = load double, ptr addrspace(1) %r53, align 8
; CHECK:      call.uni (retval0),
; CHECK-NEXT: _FOO,
; CHECK-NEXT: (
; CHECK-NEXT: param0
; CHECK-NEXT: );
%r55 = tail call double @_FOO(double %r54)
%12 = trunc i64 %r39 to i32
%r59 = mul i32 %"arg_0", %12
%r60 = add i32 %r59, 1
%r61 = sitofp i32 %r60 to double
%r65 = add i32 %r31, 1
%r66 = sitofp i32 %r65 to double
%r67 = tail call double @llvm.fma.f64(double %r55, double %r55, double %r66)
%r68 = fadd double %r67, %r61
%r71 = inttoptr i64 %"arg_2" to ptr addrspace(1)
%gep88 = getelementptr i8, ptr addrspace(1) %r71, i64 %r42
%r77 = getelementptr double, ptr addrspace(1) %gep88, i64 %11
store double %r68, ptr addrspace(1) %r77, align 8
%r80 = add i32 %r7, %"$$i_l40_0_t23.0"
%r85 = icmp slt i32 %r80, %r16
br i1 %r85, label %"BB1692", label %"EXIT_BB"

"EXIT_BB": ; preds = %"BB18", %"Entry_BB"
ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

declare double @_FOO(double)

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fma.f64(double, double, double) #1

attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{ !2}

!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
