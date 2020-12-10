; RUN: llc -march=xcore < %s | FileCheck %s
; RUN: llc -march=xcore -O=0 < %s | FileCheck %s -check-prefix=PHINODE

declare i8 addrspace(1)* @llvm.xcore.getst.p1i8.p1i8(i8 addrspace(1)* %r)
declare void @llvm.xcore.msync.p1i8(i8 addrspace(1)* %r)
declare void @llvm.xcore.ssync()
declare void @llvm.xcore.mjoin.p1i8(i8 addrspace(1)* %r)
declare void @llvm.xcore.initsp.p1i8(i8 addrspace(1)* %r, i8* %value)
declare void @llvm.xcore.initpc.p1i8(i8 addrspace(1)* %r, i8* %value)
declare void @llvm.xcore.initlr.p1i8(i8 addrspace(1)* %r, i8* %value)
declare void @llvm.xcore.initcp.p1i8(i8 addrspace(1)* %r, i8* %value)
declare void @llvm.xcore.initdp.p1i8(i8 addrspace(1)* %r, i8* %value)

define i8 addrspace(1)* @test_getst(i8 addrspace(1)* %r) {
; CHECK-LABEL: test_getst:
; CHECK: getst r0, res[r0]
  %result = call i8 addrspace(1)* @llvm.xcore.getst.p1i8.p1i8(i8 addrspace(1)* %r)
  ret i8 addrspace(1)* %result
}

define void @test_ssync() {
; CHECK-LABEL: test_ssync:
; CHECK: ssync
  call void @llvm.xcore.ssync()
  ret void
}

define void @test_mjoin(i8 addrspace(1)* %r) {
; CHECK-LABEL: test_mjoin:
; CHECK: mjoin res[r0]
  call void @llvm.xcore.mjoin.p1i8(i8 addrspace(1)* %r)
  ret void
}

define void @test_initsp(i8 addrspace(1)* %t, i8* %src) {
; CHECK-LABEL: test_initsp:
; CHECK: init t[r0]:sp, r1
  call void @llvm.xcore.initsp.p1i8(i8 addrspace(1)* %t, i8* %src)
  ret void
}

define void @test_initpc(i8 addrspace(1)* %t, i8* %src) {
; CHECK-LABEL: test_initpc:
; CHECK: init t[r0]:pc, r1
  call void @llvm.xcore.initpc.p1i8(i8 addrspace(1)* %t, i8* %src)
  ret void
}

define void @test_initlr(i8 addrspace(1)* %t, i8* %src) {
; CHECK-LABEL: test_initlr:
; CHECK: init t[r0]:lr, r1
  call void @llvm.xcore.initlr.p1i8(i8 addrspace(1)* %t, i8* %src)
  ret void
}

define void @test_initcp(i8 addrspace(1)* %t, i8* %src) {
; CHECK-LABEL: test_initcp:
; CHECK: init t[r0]:cp, r1
  call void @llvm.xcore.initcp.p1i8(i8 addrspace(1)* %t, i8* %src)
  ret void
}

define void @test_initdp(i8 addrspace(1)* %t, i8* %src) {
; CHECK-LABEL: test_initdp:
; CHECK: init t[r0]:dp, r1
  call void @llvm.xcore.initdp.p1i8(i8 addrspace(1)* %t, i8* %src)
  ret void
}

@tl = thread_local global [3 x i32] zeroinitializer
@tle = external thread_local global [2 x i32]

define i32* @f_tl() {
; CHECK-LABEL: f_tl:
; CHECK: get r11, id
; CHECK: ldaw [[R0:r[0-9]]], dp[tl]
; CHECK: ldc [[R1:r[0-9]]], 8
; CHECK: ldc [[R2:r[0-9]]], 12
; r0 = id*12 + 8 + &tl
; CHECK: lmul {{r[0-9]}}, r0, r11, [[R2]], [[R0]], [[R1]]
  ret i32* getelementptr inbounds ([3 x i32], [3 x i32]* @tl, i32 0, i32 2)
}

define i32* @f_tle() {
; CHECK-LABEL: f_tle:
; CHECK: get r11, id
; CHECK: shl [[R0:r[0-9]]], r11, 3
; CHECK: ldaw [[R1:r[0-9]]], dp[tle]
; r0 = &tl + id*8
; CHECK: add r0, [[R1]], [[R0]]
  ret i32* getelementptr inbounds ([2 x i32], [2 x i32]* @tle, i32 0, i32 0)
}

define i32 @f_tlExpr () {
; CHECK-LABEL: f_tlExpr:
; CHECK: get r11, id
; CHECK: shl [[R0:r[0-9]]], r11, 3
; CHECK: ldaw [[R1:r[0-9]]], dp[tle]
; CHECK: add [[R2:r[0-9]]], [[R1]], [[R0]]
; CHECK: add r0, [[R2]], [[R2]]
  ret i32 add(
      i32 ptrtoint( i32* getelementptr inbounds ([2 x i32], [2 x i32]* @tle, i32 0, i32 0) to i32),
      i32 ptrtoint( i32* getelementptr inbounds ([2 x i32], [2 x i32]* @tle, i32 0, i32 0) to i32))
}

define void @phiNode1() {
; N.B. lowering of duplicate constexpr in a PHI node requires -O=0
; PHINODE-LABEL: phiNode1:
; PHINODE: get r11, id
; PHINODE-LABEL: .LBB11_1:
; PHINODE: get r11, id
; PHINODE: bu .LBB11_1
entry:
  br label %ConstantExpPhiNode
ConstantExpPhiNode:
  %ptr = phi i32* [ getelementptr inbounds ([3 x i32], [3 x i32]* @tl, i32 0, i32 0), %entry ],
                  [ getelementptr inbounds ([3 x i32], [3 x i32]* @tl, i32 0, i32 0), %ConstantExpPhiNode ]
  br label %ConstantExpPhiNode
exit:
  ret void
}

define void @phiNode2( i1 %bool) {
; N.B. check an extra 'Node_crit_edge' (LBB12_1) is inserted
; PHINODE-LABEL: phiNode2:
; PHINODE: bf {{r[0-9]}}, .LBB12_3
; PHINODE: bu .LBB12_1
; PHINODE-LABEL: .LBB12_1:
; PHINODE: get r11, id
; PHINODE-LABEL: .LBB12_2:
; PHINODE: get r11, id
; PHINODE: bu .LBB12_2
; PHINODE-LABEL: .LBB12_3:
entry:
  br i1 %bool, label %ConstantExpPhiNode, label %exit
ConstantExpPhiNode:
  %ptr = phi i32* [ getelementptr inbounds ([3 x i32], [3 x i32]* @tl, i32 0, i32 0), %entry ],
                  [ getelementptr inbounds ([3 x i32], [3 x i32]* @tl, i32 0, i32 0), %ConstantExpPhiNode ]
  br label %ConstantExpPhiNode
exit:
  ret void
}

; CHECK-LABEL: tl:
; CHECK: .space  96
