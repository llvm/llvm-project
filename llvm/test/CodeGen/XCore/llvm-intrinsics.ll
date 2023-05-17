; RUN: llc < %s -march=xcore | FileCheck %s
; RUN: llc < %s -march=xcore -frame-pointer=all | FileCheck %s -check-prefix=CHECKFP

declare ptr @llvm.frameaddress(i32) nounwind readnone
declare ptr @llvm.returnaddress(i32) nounwind
declare ptr @llvm.eh.dwarf.cfa(i32) nounwind
declare void @llvm.eh.return.i32(i32, ptr) nounwind
declare void @llvm.eh.unwind.init() nounwind

define ptr @FA0() nounwind {
entry:
; CHECK-LABEL: FA0
; CHECK: ldaw r0, sp[0]
; CHECK-NEXT: retsp 0
  %0 = call ptr @llvm.frameaddress(i32 0)
  ret ptr %0
}

define ptr @FA1() nounwind {
entry:
; CHECK-LABEL: FA1
; CHECK: entsp 100
; CHECK-NEXT: ldaw r0, sp[0]
; CHECK-NEXT: retsp 100
  %0 = alloca [100 x i32]
  %1 = call ptr @llvm.frameaddress(i32 0)
  ret ptr %1
}

define ptr @RA0() nounwind {
entry:
; CHECK-LABEL: RA0
; CHECK: stw lr, sp[0]
; CHECK-NEXT: ldw r0, sp[0]
; CHECK-NEXT: ldw lr, sp[0]
; CHECK-NEXT: retsp 0
  %0 = call ptr @llvm.returnaddress(i32 0)
  ret ptr %0
}

define ptr @RA1() nounwind {
entry:
; CHECK-LABEL: RA1
; CHECK: entsp 100
; CHECK-NEXT: ldw r0, sp[100]
; CHECK-NEXT: retsp 100
  %0 = alloca [100 x i32]
  %1 = call ptr @llvm.returnaddress(i32 0)
  ret ptr %1
}

; test FRAME_TO_ARGS_OFFSET lowering
define ptr @FTAO0() nounwind {
entry:
; CHECK-LABEL: FTAO0
; CHECK: ldc r0, 0
; CHECK-NEXT: ldaw r1, sp[0]
; CHECK-NEXT: add r0, r1, r0
; CHECK-NEXT: retsp 0
  %0 = call ptr @llvm.eh.dwarf.cfa(i32 0)
  ret ptr %0
}

define ptr @FTAO1() nounwind {
entry:
; CHECK-LABEL: FTAO1
; CHECK: entsp 100
; CHECK-NEXT: ldc r0, 400
; CHECK-NEXT: ldaw r1, sp[0]
; CHECK-NEXT: add r0, r1, r0
; CHECK-NEXT: retsp 100
  %0 = alloca [100 x i32]
  %1 = call ptr @llvm.eh.dwarf.cfa(i32 0)
  ret ptr %1
}

define ptr @EH0(i32 %offset, ptr %handler) {
entry:
; CHECK-LABEL: EH0
; CHECK: entsp 2
; CHECK: .cfi_def_cfa_offset 8
; CHECK: .cfi_offset 15, 0
; CHECK: .cfi_offset 1, -8
; CHECK: .cfi_offset 0, -4
; CHECK: ldc r2, 8
; CHECK-NEXT: ldaw r3, sp[0]
; CHECK-NEXT: add r2, r3, r2
; CHECK-NEXT: add r2, r2, r0
; CHECK-NEXT: mov r3, r1
; CHECK-NEXT: ldw r1, sp[0]
; CHECK-NEXT: ldw r0, sp[1]
; CHECK-NEXT: set sp, r2
; CHECK-NEXT: bau r3
  call void @llvm.eh.return.i32(i32 %offset, ptr %handler)
  unreachable
}

declare void @foo(...)
define ptr @EH1(i32 %offset, ptr %handler) {
entry:
; CHECK-LABEL: EH1
; CHECK: entsp 5
; CHECK: .cfi_def_cfa_offset 20
; CHECK: .cfi_offset 15, 0
; CHECK: .cfi_offset 1, -16
; CHECK: .cfi_offset 0, -12
; CHECK: stw r4, sp[4]
; CHECK: .cfi_offset 4, -4
; CHECK: stw r5, sp[3]
; CHECK: .cfi_offset 5, -8
; CHECK: mov r4, r1
; CHECK-NEXT: mov r5, r0
; CHECK-NEXT: bl foo
; CHECK-NEXT: ldc r0, 20
; CHECK-NEXT: ldaw r1, sp[0]
; CHECK-NEXT: add r0, r1, r0
; CHECK-NEXT: add r2, r0, r5
; CHECK-NEXT: mov r3, r4
; CHECK-NEXT: ldw r5, sp[3]
; CHECK-NEXT: ldw r4, sp[4]
; CHECK-NEXT: ldw r1, sp[1]
; CHECK-NEXT: ldw r0, sp[2]
; CHECK-NEXT: set sp, r2
; CHECK-NEXT: bau r3
  call void (...) @foo()
  call void @llvm.eh.return.i32(i32 %offset, ptr %handler)
  unreachable
}

@offset = external constant i32
@handler = external constant i8
define ptr @EH2(i32 %r0, i32 %r1, i32 %r2, i32 %r3) {
entry:
; CHECK-LABEL: EH2
; CHECK: entsp 3
; CHECK: bl foo
; CHECK-NEXT: ldw r0, dp[offset]
; CHECK-NEXT: ldc r1, 12
; CHECK-NEXT: ldaw r2, sp[0]
; CHECK-NEXT: add r1, r2, r1
; CHECK-NEXT: add r2, r1, r0
; CHECK-NEXT: ldaw r3, dp[handler]
; CHECK-NEXT: ldw r1, sp[1]
; CHECK-NEXT: ldw r0, sp[2]
; CHECK-NEXT: set sp, r2
; CHECK-NEXT: bau r3
  call void (...) @foo()
  %0 = load i32, ptr @offset
  call void @llvm.eh.return.i32(i32 %0, ptr @handler)
  unreachable
}


; FP: spill FP+SR+R0:1+R4:9 = entsp 2+2+6
; But we dont actually spill or restore R0:1
; CHECKFP-LABEL: Unwind0:
; CHECKFP: entsp 10
; CHECKFP: stw r10, sp[1]
; CHECKFP: ldaw r10, sp[0]
; CHECKFP: stw r4, r10[9]
; CHECKFP: stw r5, r10[8]
; CHECKFP: stw r6, r10[7]
; CHECKFP: stw r7, r10[6]
; CHECKFP: stw r8, r10[5]
; CHECKFP: stw r9, r10[4]
; CHECKFP: ldw r9, r10[4]
; CHECKFP: ldw r8, r10[5]
; CHECKFP: ldw r7, r10[6]
; CHECKFP: ldw r6, r10[7]
; CHECKFP: ldw r5, r10[8]
; CHECKFP: ldw r4, r10[9]
; CHECKFP: set sp, r10
; CHECKFP: ldw r10, sp[1]
; CHECKFP: retsp 10
;
; !FP: spill R0:1+R4:10 = entsp 2+7
; But we dont actually spill or restore R0:1
; CHECK-LABEL: Unwind0:
; CHECK: entsp 9
; CHECK: stw r4, sp[8]
; CHECK: stw r5, sp[7]
; CHECK: stw r6, sp[6]
; CHECK: stw r7, sp[5]
; CHECK: stw r8, sp[4]
; CHECK: stw r9, sp[3]
; CHECK: stw r10, sp[2]
; CHECK: ldw r10, sp[2]
; CHECK: ldw r9, sp[3]
; CHECK: ldw r8, sp[4]
; CHECK: ldw r7, sp[5]
; CHECK: ldw r6, sp[6]
; CHECK: ldw r5, sp[7]
; CHECK: ldw r4, sp[8]
; CHECK: retsp 9
define void @Unwind0() {
  call void @llvm.eh.unwind.init()
  ret void
}


; FP: spill FP+SR+R0:1+R4:9+LR = entsp 2+2+6 + extsp 1
; But we dont actually spill or restore R0:1
; CHECKFP-LABEL: Unwind1:
; CHECKFP: entsp 10
; CHECKFP: stw r10, sp[1]
; CHECKFP: ldaw r10, sp[0]
; CHECKFP: stw r4, r10[9]
; CHECKFP: stw r5, r10[8]
; CHECKFP: stw r6, r10[7]
; CHECKFP: stw r7, r10[6]
; CHECKFP: stw r8, r10[5]
; CHECKFP: stw r9, r10[4]
; CHECKFP: extsp 1
; CHECKFP: bl foo
; CHECKFP: ldaw sp, sp[1]
; CHECKFP: ldw r9, r10[4]
; CHECKFP: ldw r8, r10[5]
; CHECKFP: ldw r7, r10[6]
; CHECKFP: ldw r6, r10[7]
; CHECKFP: ldw r5, r10[8]
; CHECKFP: ldw r4, r10[9]
; CHECKFP: set sp, r10
; CHECKFP: ldw r10, sp[1]
; CHECKFP: retsp 10
;
; !FP: spill R0:1+R4:10+LR = entsp 2+7+1
; But we dont actually spill or restore R0:1
; CHECK-LABEL: Unwind1:
; CHECK: entsp 10
; CHECK: stw r4, sp[9]
; CHECK: stw r5, sp[8]
; CHECK: stw r6, sp[7]
; CHECK: stw r7, sp[6]
; CHECK: stw r8, sp[5]
; CHECK: stw r9, sp[4]
; CHECK: stw r10, sp[3]
; CHECK: bl foo
; CHECK: ldw r10, sp[3]
; CHECK: ldw r9, sp[4]
; CHECK: ldw r8, sp[5]
; CHECK: ldw r7, sp[6]
; CHECK: ldw r6, sp[7]
; CHECK: ldw r5, sp[8]
; CHECK: ldw r4, sp[9]
; CHECK: retsp 10
define void @Unwind1() {
  call void (...) @foo()
  call void @llvm.eh.unwind.init()
  ret void
}

; FP: spill FP+SR+R0:1+R4:9 = entsp 2+2+6
; We dont spill R0:1
; We only restore R0:1 during eh.return
; CHECKFP-LABEL: UnwindEH:
; CHECKFP: entsp 10
; CHECKFP: .cfi_def_cfa_offset 40
; CHECKFP: .cfi_offset 15, 0
; CHECKFP: stw r10, sp[1]
; CHECKFP: .cfi_offset 10, -36
; CHECKFP: ldaw r10, sp[0]
; CHECKFP: .cfi_def_cfa_register 10
; CHECKFP: .cfi_offset 1, -32
; CHECKFP: .cfi_offset 0, -28
; CHECKFP: stw r4, r10[9]
; CHECKFP: .cfi_offset 4, -4
; CHECKFP: stw r5, r10[8]
; CHECKFP: .cfi_offset 5, -8
; CHECKFP: stw r6, r10[7]
; CHECKFP: .cfi_offset 6, -12
; CHECKFP: stw r7, r10[6]
; CHECKFP: .cfi_offset 7, -16
; CHECKFP: stw r8, r10[5]
; CHECKFP: .cfi_offset 8, -20
; CHECKFP: stw r9, r10[4]
; CHECKFP: .cfi_offset 9, -24
; CHECKFP: bt r0, .LBB{{[0-9_]+}}
; CHECKFP: ldw r9, r10[4]
; CHECKFP-NEXT: ldw r8, r10[5]
; CHECKFP-NEXT: ldw r7, r10[6]
; CHECKFP-NEXT: ldw r6, r10[7]
; CHECKFP-NEXT: ldw r5, r10[8]
; CHECKFP-NEXT: ldw r4, r10[9]
; CHECKFP-NEXT: set sp, r10
; CHECKFP-NEXT: ldw r10, sp[1]
; CHECKFP-NEXT: retsp 10
; CHECKFP: .LBB{{[0-9_]+}}
; CHECKFP-NEXT: ldc r2, 40
; CHECKFP-NEXT: add r2, r10, r2
; CHECKFP-NEXT: add r2, r2, r0
; CHECKFP-NEXT: mov r3, r1
; CHECKFP-NEXT: ldw r9, r10[4]
; CHECKFP-NEXT: ldw r8, r10[5]
; CHECKFP-NEXT: ldw r7, r10[6]
; CHECKFP-NEXT: ldw r6, r10[7]
; CHECKFP-NEXT: ldw r5, r10[8]
; CHECKFP-NEXT: ldw r4, r10[9]
; CHECKFP-NEXT: ldw r1, sp[2]
; CHECKFP-NEXT: ldw r0, sp[3]
; CHECKFP-NEXT: set sp, r2
; CHECKFP-NEXT: bau r3
;
; !FP: spill R0:1+R4:10 = entsp 2+7
; We dont spill R0:1
; We only restore R0:1 during eh.return
; CHECK-LABEL: UnwindEH:
; CHECK: entsp 9
; CHECK: .cfi_def_cfa_offset 36
; CHECK: .cfi_offset 15, 0
; CHECK: .cfi_offset 1, -36
; CHECK: .cfi_offset 0, -32
; CHECK: stw r4, sp[8]
; CHECK: .cfi_offset 4, -4
; CHECK: stw r5, sp[7]
; CHECK: .cfi_offset 5, -8
; CHECK: stw r6, sp[6]
; CHECK: .cfi_offset 6, -12
; CHECK: stw r7, sp[5]
; CHECK: .cfi_offset 7, -16
; CHECK: stw r8, sp[4]
; CHECK: .cfi_offset 8, -20
; CHECK: stw r9, sp[3]
; CHECK: .cfi_offset 9, -24
; CHECK: stw r10, sp[2]
; CHECK: .cfi_offset 10, -28
; CHECK: bt r0, .LBB{{[0-9_]+}}
; CHECK: ldw r10, sp[2]
; CHECK-NEXT: ldw r9, sp[3]
; CHECK-NEXT: ldw r8, sp[4]
; CHECK-NEXT: ldw r7, sp[5]
; CHECK-NEXT: ldw r6, sp[6]
; CHECK-NEXT: ldw r5, sp[7]
; CHECK-NEXT: ldw r4, sp[8]
; CHECK-NEXT: retsp 9
; CHECK: .LBB{{[0-9_]+}}
; CHECK-NEXT: ldc r2, 36
; CHECK-NEXT: ldaw r3, sp[0]
; CHECK-NEXT: add r2, r3, r2
; CHECK-NEXT: add r2, r2, r0
; CHECK-NEXT: mov r3, r1
; CHECK-NEXT: ldw r10, sp[2]
; CHECK-NEXT: ldw r9, sp[3]
; CHECK-NEXT: ldw r8, sp[4]
; CHECK-NEXT: ldw r7, sp[5]
; CHECK-NEXT: ldw r6, sp[6]
; CHECK-NEXT: ldw r5, sp[7]
; CHECK-NEXT: ldw r4, sp[8]
; CHECK-NEXT: ldw r1, sp[0]
; CHECK-NEXT: ldw r0, sp[1]
; CHECK-NEXT: set sp, r2
; CHECK-NEXT: bau r3
define void @UnwindEH(i32 %offset, ptr %handler) {
  call void @llvm.eh.unwind.init()
  %cmp = icmp eq i32 %offset, 0
  br i1 %cmp, label %normal, label %eh
eh:
  call void @llvm.eh.return.i32(i32 %offset, ptr %handler)
  unreachable
normal:
  ret void
}
