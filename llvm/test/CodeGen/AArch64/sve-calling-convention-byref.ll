; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -stop-after=finalize-isel < %s | FileCheck %s --check-prefixes=CHECK,LINUX
; RUN: llc -mtriple=aarch64-apple-darwin -mattr=+sve -stop-after=finalize-isel < %s | FileCheck %s --check-prefixes=CHECK,DARWIN

; Test that z8 and z9, passed in by reference, are correctly loaded from x0 and x1.
; i.e. z0 =  %z0
;         :
;      z7 =  %z7
;      x0 = &%z8
;      x1 = &%z9
define aarch64_sve_vector_pcs <vscale x 4 x i32> @callee_with_many_sve_arg(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z3, <vscale x 4 x i32> %z4, <vscale x 4 x i32> %z5, <vscale x 4 x i32> %z6, <vscale x 4 x i32> %z7, <vscale x 4 x i32> %z8, <vscale x 4 x i32> %z9) {
; CHECK: name: callee_with_many_sve_arg
; CHECK-DAG: [[BASE:%[0-9]+]]:gpr64common = COPY $x1
; CHECK-DAG: [[RES:%[0-9]+]]:zpr = LDR_ZXI [[BASE]], 0
; CHECK-DAG: $z0 = COPY [[RES]]
; CHECK:     RET_ReallyLR implicit $z0
  ret <vscale x 4 x i32> %z9
}

; Test that z8 and z9 are passed by reference.
define aarch64_sve_vector_pcs <vscale x 4 x i32> @caller_with_many_sve_arg(<vscale x 4 x i32> %z) {
; CHECK: name: caller_with_many_sve_arg
; CHECK: stack:
; CHECK:      - { id: 0, name: '', type: default, offset: 0, size: 16, alignment: 16,
; CHECK-NEXT:     stack-id: scalable-vector
; CHECK:      - { id: 1, name: '', type: default, offset: 0, size: 16, alignment: 16,
; CHECK-NEXT:     stack-id: scalable-vector
; CHECK-DAG:  STR_ZXI %{{[0-9]+}}, %stack.1, 0
; CHECK-DAG:  STR_ZXI %{{[0-9]+}}, %stack.0, 0
; CHECK-DAG:  [[BASE2:%[0-9]+]]:gpr64sp = ADDXri %stack.1, 0
; CHECK-DAG:  [[BASE1:%[0-9]+]]:gpr64sp = ADDXri %stack.0, 0
; CHECK-DAG:  $x0 = COPY [[BASE1]]
; CHECK-DAG:  $x1 = COPY [[BASE2]]
; CHECK-NEXT: BL @callee_with_many_sve_arg
; CHECK:      RET_ReallyLR implicit $z0
  %ret = call aarch64_sve_vector_pcs <vscale x 4 x i32> @callee_with_many_sve_arg(<vscale x 4 x i32> %z, <vscale x 4 x i32> %z, <vscale x 4 x i32> %z, <vscale x 4 x i32> %z, <vscale x 4 x i32> %z, <vscale x 4 x i32> %z, <vscale x 4 x i32> %z, <vscale x 4 x i32> %z, <vscale x 4 x i32> %z, <vscale x 4 x i32> %z)
  ret <vscale x 4 x i32> %ret
}

; Test that p4 and p5, passed in by reference, are correctly loaded from register x0 and x1.
; i.e. p0 =  %p0
;         :
;      p3 =  %p3
;      x0 = &%p4
;      x1 = &%p5
define aarch64_sve_vector_pcs <vscale x 16 x i1> @callee_with_many_svepred_arg(<vscale x 16 x i1> %p0, <vscale x 16 x i1> %p1, <vscale x 16 x i1> %p2, <vscale x 16 x i1> %p3, <vscale x 16 x i1> %p4, <vscale x 16 x i1> %p5) {
; CHECK: name: callee_with_many_svepred_arg
; CHECK-DAG: [[BASE:%[0-9]+]]:gpr64common = COPY $x1
; CHECK-DAG: [[RES:%[0-9]+]]:ppr = LDR_PXI [[BASE]], 0
; CHECK-DAG: $p0 = COPY [[RES]]
; CHECK:     RET_ReallyLR implicit $p0
  ret <vscale x 16 x i1> %p5
}

; Test that p4 and p5 are passed by reference.
define aarch64_sve_vector_pcs <vscale x 16 x i1> @caller_with_many_svepred_arg(<vscale x 16 x i1> %p) {
; CHECK: name: caller_with_many_svepred_arg
; CHECK: stack:
; CHECK:      - { id: 0, name: '', type: default, offset: 0, size: 2, alignment: 2,
; CHECK-NEXT:     stack-id: scalable-predicate-vector
; CHECK:      - { id: 1, name: '', type: default, offset: 0, size: 2, alignment: 2,
; CHECK-NEXT:     stack-id: scalable-predicate-vector
; CHECK-DAG: STR_PXI %{{[0-9]+}}, %stack.0, 0
; CHECK-DAG: STR_PXI %{{[0-9]+}}, %stack.1, 0
; CHECK-DAG: [[BASE1:%[0-9]+]]:gpr64sp = ADDXri %stack.0, 0
; CHECK-DAG: [[BASE2:%[0-9]+]]:gpr64sp = ADDXri %stack.1, 0
; CHECK-DAG: $x0 = COPY [[BASE1]]
; CHECK-DAG: $x1 = COPY [[BASE2]]
; CHECK-NEXT: BL @callee_with_many_svepred_arg
; CHECK:     RET_ReallyLR implicit $p0
  %ret = call aarch64_sve_vector_pcs <vscale x 16 x i1> @callee_with_many_svepred_arg(<vscale x 16 x i1> %p, <vscale x 16 x i1> %p, <vscale x 16 x i1> %p, <vscale x 16 x i1> %p, <vscale x 16 x i1> %p, <vscale x 16 x i1> %p)
  ret <vscale x 16 x i1> %ret
}

; Test that arg2 is passed through x0, i.e., x0 = &%arg2; and return values are loaded from x0:
;     P0 = ldr [x0]
define aarch64_sve_vector_pcs <vscale x 16 x i1> @callee_with_svepred_arg_4xv16i1_1xv16i1([4 x <vscale x 16 x i1>] %arg1, [1 x <vscale x 16 x i1>] %arg2) {
; CHECK: name: callee_with_svepred_arg_4xv16i1_1xv16i1
; CHECK:    [[BASE:%[0-9]+]]:gpr64common = COPY $x0
; CHECK:    [[PRED0:%[0-9]+]]:ppr = LDR_PXI [[BASE]], 0 :: (load (<vscale x 1 x s16>))
; CHECK:    $p0 = COPY [[PRED0]]
; CHECK:    RET_ReallyLR implicit $p0
  %res = extractvalue [1 x <vscale x 16 x i1>] %arg2, 0
  ret <vscale x 16 x i1> %res
}

; Test that arg1 is stored to the stack from p0; and the stack location is passed throuch x0 to setup the call:
;     str P0, [stack_loc_for_args]
;     x0 = stack_loc_for_args
define aarch64_sve_vector_pcs <vscale x 16 x i1> @caller_with_svepred_arg_1xv16i1_4xv16i1([1 x <vscale x 16 x i1>] %arg1, [4 x <vscale x 16 x i1>] %arg2) {
; CHECK: name: caller_with_svepred_arg_1xv16i1_4xv16i1
; CHECK: stack:
; CHECK:      - { id: 0, name: '', type: default, offset: 0, size: 2, alignment: 2,
; CHECK-NEXT:     stack-id: scalable-predicate-vector,
; CHECK:    [[PRED0:%[0-9]+]]:ppr = COPY $p0
; CHECK:    ADJCALLSTACKDOWN 0, 0, implicit-def dead $sp, implicit $sp
; CHECK:    STR_PXI [[PRED0]], %stack.0, 0 :: (store (<vscale x 1 x s16>) into %stack.0)
; CHECK:    [[STACK:%[0-9]+]]:gpr64sp = ADDXri %stack.0, 0, 0
; CHECK:    $x0 = COPY [[STACK]]
; LINUX:    BL @callee_with_svepred_arg_4xv16i1_1xv16i1, csr_aarch64_sve_aapcs, implicit-def dead $lr, implicit $sp, implicit $p0, implicit $p1, implicit $p2, implicit $p3, implicit $x0, implicit-def $sp, implicit-def $p0
; DARWIN:   BL @callee_with_svepred_arg_4xv16i1_1xv16i1, csr_darwin_aarch64_sve_aapcs, implicit-def dead $lr, implicit $sp, implicit $p0, implicit $p1, implicit $p2, implicit $p3, implicit $x0, implicit-def $sp, implicit-def $p0
; CHECK:    ADJCALLSTACKUP 0, 0, implicit-def dead $sp, implicit $sp
  %res = call <vscale x 16 x i1> @callee_with_svepred_arg_4xv16i1_1xv16i1([4 x <vscale x 16 x i1>] %arg2, [1 x <vscale x 16 x i1>] %arg1)
  ret <vscale x 16 x i1> %res
}

; Test that arg2 is passed through x0, i.e., x0 = &%arg2; and return values are loaded from x0:
;     P0 = ldr [x0]
;     P1 = ldr [x0 +   sizeof(Px)]
;     P2 = ldr [x0 + 2*sizeof(Px)]
;     P3 = ldr [x0 + 3*sizeof(Px)]
define aarch64_sve_vector_pcs [4 x <vscale x 16 x i1>] @callee_with_svepred_arg_4xv16i1_4xv16i1([4 x <vscale x 16 x i1>] %arg1, [4 x <vscale x 16 x i1>] %arg2) {
; CHECK: name: callee_with_svepred_arg_4xv16i1_4xv16i1
; CHECK:    [[BASE:%[0-9]+]]:gpr64common = COPY $x0
; CHECK:    [[OFFSET1:%[0-9]+]]:gpr64 = CNTD_XPiI 31, 1, implicit $vg
; CHECK:    [[ADDR1:%[0-9]+]]:gpr64common = nuw ADDXrr [[BASE]], killed [[OFFSET1]]
; CHECK:    [[PRED1:%[0-9]+]]:ppr = LDR_PXI killed [[ADDR1]], 0 :: (load (<vscale x 1 x s16>))
; CHECK:    [[OFFSET2:%[0-9]+]]:gpr64 = CNTW_XPiI 31, 1, implicit $vg
; CHECK:    [[ADDR2:%[0-9]+]]:gpr64common = ADDXrr [[BASE]], killed [[OFFSET2]]
; CHECK:    [[PRED2:%[0-9]+]]:ppr = LDR_PXI killed [[ADDR2]], 0 :: (load (<vscale x 1 x s16>))
; CHECK:    [[OFFSET3:%[0-9]+]]:gpr64 = CNTD_XPiI 31, 3, implicit $vg
; CHECK:    [[ADDR3:%[0-9]+]]:gpr64common = ADDXrr [[BASE]], killed [[OFFSET3]]
; CHECK:    [[PRED3:%[0-9]+]]:ppr = LDR_PXI killed [[ADDR3]], 0 :: (load (<vscale x 1 x s16>))
; CHECK:    [[PRED0:%[0-9]+]]:ppr = LDR_PXI [[BASE]], 0 :: (load (<vscale x 1 x s16>))
; CHECK:    $p0 = COPY [[PRED0]]
; CHECK:    $p1 = COPY [[PRED1]]
; CHECK:    $p2 = COPY [[PRED2]]
; CHECK:    $p3 = COPY [[PRED3]]
; CHECK:    RET_ReallyLR implicit $p0, implicit $p1, implicit $p2, implicit $p3
  ret [4 x <vscale x 16 x i1>] %arg2
}

; Test that arg1 is stored to the stack from p0~p3; and the stack location is passed throuch x0 to setup the call:
;     str P0, [stack_loc_for_args]
;     str P1, [stack_loc_for_args +   sizeof(Px)]
;     str P2, [stack_loc_for_args + 2*sizeof(Px)]
;     str P3, [stack_loc_for_args + 3*sizeof(Px)]
;     x0 = stack_loc_for_args
define [4 x <vscale x 16 x i1>] @caller_with_svepred_arg_4xv16i1_4xv16i1([4 x <vscale x 16 x i1>] %arg1, [4 x <vscale x 16 x i1>] %arg2) {
; CHECK: name: caller_with_svepred_arg_4xv16i1_4xv16i1
; CHECK: stack:
; CHECK:      - { id: 0, name: '', type: default, offset: 0, size: 8, alignment: 2,
; CHECK-NEXT:     stack-id: scalable-predicate-vector,
; CHECK:    [[PRED3:%[0-9]+]]:ppr = COPY $p3
; CHECK:    [[PRED2:%[0-9]+]]:ppr = COPY $p2
; CHECK:    [[PRED1:%[0-9]+]]:ppr = COPY $p1
; CHECK:    [[PRED0:%[0-9]+]]:ppr = COPY $p0
; CHECK:    [[OFFSET1:%[0-9]+]]:gpr64 = CNTD_XPiI 31, 1, implicit $vg
; CHECK:    [[OFFSET2:%[0-9]+]]:gpr64 = CNTW_XPiI 31, 1, implicit $vg
; CHECK:    [[OFFSET3:%[0-9]+]]:gpr64 = CNTD_XPiI 31, 3, implicit $vg
; CHECK:    [[STACK:%[0-9]+]]:gpr64common = ADDXri %stack.0, 0, 0
; CHECK:    [[ADDR3:%[0-9]+]]:gpr64common = ADDXrr [[STACK]], [[OFFSET3]]
; CHECK:    ADJCALLSTACKDOWN 0, 0, implicit-def dead $sp, implicit $sp
; CHECK:    STR_PXI [[PRED3]], killed [[ADDR3]], 0 :: (store (<vscale x 1 x s16>))
; CHECK:    [[ADDR2:%[0-9]+]]:gpr64common = ADDXrr [[STACK]], [[OFFSET2]]
; CHECK:    STR_PXI [[PRED2]], killed [[ADDR2]], 0 :: (store (<vscale x 1 x s16>))
; CHECK:    [[ADDR1:%[0-9]+]]:gpr64common = nuw ADDXrr [[STACK]], [[OFFSET1]]
; CHECK:    STR_PXI [[PRED1]], killed [[ADDR1]], 0 :: (store (<vscale x 1 x s16>))
; CHECK:    STR_PXI [[PRED0]], %stack.0, 0 :: (store (<vscale x 1 x s16>) into %stack.0)
; CHECK:    $x0 = COPY [[STACK]]
; LINUX:    BL @callee_with_svepred_arg_4xv16i1_4xv16i1, csr_aarch64_sve_aapcs, implicit-def dead $lr, implicit $sp, implicit $p0, implicit $p1, implicit $p2, implicit $p3, implicit $x0, implicit-def $sp, implicit-def $p0, implicit-def $p1, implicit-def $p2, implicit-def $p3
; DARWIN:   BL @callee_with_svepred_arg_4xv16i1_4xv16i1, csr_darwin_aarch64_sve_aapcs, implicit-def dead $lr, implicit $sp, implicit $p0, implicit $p1, implicit $p2, implicit $p3, implicit $x0, implicit-def $sp, implicit-def $p0, implicit-def $p1, implicit-def $p2, implicit-def $p3
; CHECK:    ADJCALLSTACKUP 0, 0, implicit-def dead $sp, implicit $sp
  %res = call [4 x <vscale x 16 x i1>] @callee_with_svepred_arg_4xv16i1_4xv16i1([4 x <vscale x 16 x i1>] %arg2, [4 x <vscale x 16 x i1>] %arg1)
  ret [4 x <vscale x 16 x i1>] %res
}

; Test that arg2 is passed through x0, i.e., x0 = &%arg2; and return values are loaded from x0:
;     P0 = ldr [x0]
;     P1 = ldr [x0 +   sizeof(Px)]
;     P2 = ldr [x0 + 2*sizeof(Px)]
;     P3 = ldr [x0 + 3*sizeof(Px)]
define aarch64_sve_vector_pcs [2 x <vscale x 32 x i1>] @callee_with_svepred_arg_1xv16i1_2xv32i1([1 x <vscale x 16 x i1>] %arg1, [2 x <vscale x 32 x i1>] %arg2) {
; CHECK: name: callee_with_svepred_arg_1xv16i1_2xv32i1
; CHECK:    [[BASE:%[0-9]+]]:gpr64common = COPY $x0
; CHECK:    [[OFFSET1:%[0-9]+]]:gpr64 = CNTD_XPiI 31, 1, implicit $vg
; CHECK:    [[ADDR1:%[0-9]+]]:gpr64common = nuw ADDXrr [[BASE]], killed [[OFFSET1]]
; CHECK:    [[PRED1:%[0-9]+]]:ppr = LDR_PXI killed [[ADDR1]], 0 :: (load (<vscale x 1 x s16>))
; CHECK:    [[OFFSET2:%[0-9]+]]:gpr64 = CNTW_XPiI 31, 1, implicit $vg
; CHECK:    [[ADDR2:%[0-9]+]]:gpr64common = ADDXrr [[BASE]], killed [[OFFSET2]]
; CHECK:    [[PRED2:%[0-9]+]]:ppr = LDR_PXI killed [[ADDR2]], 0 :: (load (<vscale x 1 x s16>))
; CHECK:    [[OFFSET3:%[0-9]+]]:gpr64 = CNTD_XPiI 31, 3, implicit $vg
; CHECK:    [[ADDR3:%[0-9]+]]:gpr64common = ADDXrr [[BASE]], killed [[OFFSET3]]
; CHECK:    [[PRED3:%[0-9]+]]:ppr = LDR_PXI killed [[ADDR3]], 0 :: (load (<vscale x 1 x s16>))
; CHECK:    [[PRED0:%[0-9]+]]:ppr = LDR_PXI [[BASE]], 0 :: (load (<vscale x 1 x s16>))
; CHECK:    $p0 = COPY [[PRED0]]
; CHECK:    $p1 = COPY [[PRED1]]
; CHECK:    $p2 = COPY [[PRED2]]
; CHECK:    $p3 = COPY [[PRED3]]
; CHECK:    RET_ReallyLR implicit $p0, implicit $p1, implicit $p2, implicit $p3
  ret [2 x <vscale x 32 x i1>] %arg2
}

; Test that arg1 is stored to the stack from p0~p3; and the stack location is passed throuch x0 to setup the call:
;     str P0, [stack_loc_for_args]
;     str P1, [stack_loc_for_args +   sizeof(Px)]
;     str P2, [stack_loc_for_args + 2*sizeof(Px)]
;     str P3, [stack_loc_for_args + 3*sizeof(Px)]
;     x0 = stack_loc_for_args
define [2 x <vscale x 32 x i1>] @caller_with_svepred_arg_2xv32i1_1xv16i1([2 x <vscale x 32 x i1>] %arg1, [1 x <vscale x 16 x i1>] %arg2) {
; CHECK: name: caller_with_svepred_arg_2xv32i1_1xv16i1
; CHECK: stack:
; CHECK:      - { id: 0, name: '', type: default, offset: 0, size: 8, alignment: 2,
; CHECK-NEXT:     stack-id: scalable-predicate-vector,
; CHECK:    [[PRED3:%[0-9]+]]:ppr = COPY $p3
; CHECK:    [[PRED2:%[0-9]+]]:ppr = COPY $p2
; CHECK:    [[PRED1:%[0-9]+]]:ppr = COPY $p1
; CHECK:    [[PRED0:%[0-9]+]]:ppr = COPY $p0
; CHECK:    [[OFFSET3:%[0-9]+]]:gpr64 = CNTD_XPiI 31, 3, implicit $vg
; CHECK:    [[STACK:%[0-9]+]]:gpr64common = ADDXri %stack.0, 0, 0
; CHECK:    [[ADDR3:%[0-9]+]]:gpr64common = ADDXrr [[STACK]], killed [[OFFSET3]]
; CHECK:    ADJCALLSTACKDOWN 0, 0, implicit-def dead $sp, implicit $sp
; CHECK:    STR_PXI [[PRED3]], killed [[ADDR3]], 0 :: (store (<vscale x 1 x s16>))
; CHECK:    [[OFFSET2:%[0-9]+]]:gpr64 = CNTW_XPiI 31, 1, implicit $vg
; CHECK:    [[ADDR2:%[0-9]+]]:gpr64common = ADDXrr [[STACK]], killed [[OFFSET2]]
; CHECK:    STR_PXI [[PRED2]], killed [[ADDR2]], 0 :: (store (<vscale x 1 x s16>))
; CHECK:    [[OFFSET1:%[0-9]+]]:gpr64 = CNTD_XPiI 31, 1, implicit $vg
; CHECK:    [[ADDR1:%[0-9]+]]:gpr64common = nuw ADDXrr [[STACK]], killed [[OFFSET1]]
; CHECK:    STR_PXI [[PRED1]], killed [[ADDR1]], 0 :: (store (<vscale x 1 x s16>))
; CHECK:    STR_PXI [[PRED0]], %stack.0, 0 :: (store (<vscale x 1 x s16>) into %stack.0)
; CHECK:    $x0 = COPY [[STACK]]
; LINUX:    BL @callee_with_svepred_arg_1xv16i1_2xv32i1, csr_aarch64_sve_aapcs, implicit-def dead $lr, implicit $sp, implicit $p0, implicit $x0, implicit-def $sp, implicit-def $p0, implicit-def $p1, implicit-def $p2, implicit-def $p3
; DARWIN:   BL @callee_with_svepred_arg_1xv16i1_2xv32i1, csr_darwin_aarch64_sve_aapcs, implicit-def dead $lr, implicit $sp, implicit $p0, implicit $x0, implicit-def $sp, implicit-def $p0, implicit-def $p1, implicit-def $p2, implicit-def $p3
; CHECK:    ADJCALLSTACKUP 0, 0, implicit-def dead $sp, implicit $sp
  %res = call [2 x <vscale x 32 x i1>] @callee_with_svepred_arg_1xv16i1_2xv32i1([1 x <vscale x 16 x i1>] %arg2, [2 x <vscale x 32 x i1>] %arg1)
  ret [2 x <vscale x 32 x i1>] %res
}

; Test that arg1 and arg3 are passed via P0~P3, arg1 is passed indirectly through address on stack in x0
define aarch64_sve_vector_pcs [4 x <vscale x 16 x i1>] @callee_with_svepred_arg_2xv16i1_4xv16i1_2xv16i1([2 x <vscale x 16 x i1>] %arg1, [4 x <vscale x 16 x i1>] %arg2, [2 x <vscale x 16 x i1>] %arg3) nounwind {
; CHECK: name: callee_with_svepred_arg_2xv16i1_4xv16i1_2xv16i1
; CHECK:    [[P3:%[0-9]+]]:ppr = COPY $p3
; CHECK:    [[P2:%[0-9]+]]:ppr = COPY $p2
; CHECK:    [[X0:%[0-9]+]]:gpr64common = COPY $x0
; CHECK:    [[P1:%[0-9]+]]:ppr = COPY $p1
; CHECK:    [[P0:%[0-9]+]]:ppr = COPY $p0
; CHECK:    [[OFFSET3:%[0-9]+]]:gpr64 = CNTD_XPiI 31, 3, implicit $vg
; CHECK:    [[ADDR3:%[0-9]+]]:gpr64common = ADDXrr [[X0]], killed [[OFFSET3]]
; CHECK:    [[P7:%[0-9]+]]:ppr = LDR_PXI killed [[ADDR3]], 0 :: (load (<vscale x 1 x s16>))
; CHECK:    [[OFFSET2:%[0-9]+]]:gpr64 = CNTW_XPiI 31, 1, implicit $vg
; CHECK:    [[ADDR2:%[0-9]+]]:gpr64common = ADDXrr [[X0]], killed [[OFFSET2]]
; CHECK:    [[P6:%[0-9]+]]:ppr = LDR_PXI killed [[ADDR2]], 0 :: (load (<vscale x 1 x s16>))
; CHECK:    [[OFFSET1:%[0-9]+]]:gpr64 = CNTD_XPiI 31, 1, implicit $vg
; CHECK:    [[ADDR1:%[0-9]+]]:gpr64common = nuw ADDXrr [[X0]], killed [[OFFSET1]]
; CHECK:    [[P5:%[0-9]+]]:ppr = LDR_PXI killed [[ADDR1]], 0 :: (load (<vscale x 1 x s16>))
; CHECK:    [[P4:%[0-9]+]]:ppr = LDR_PXI [[X0]], 0 :: (load (<vscale x 1 x s16>))
; CHECK:    [[RES0:%[0-9]+]]:ppr = AND_PPzPP [[P0]], [[P0]], killed [[P4]]
; CHECK:    [[RES1:%[0-9]+]]:ppr = AND_PPzPP [[P1]], [[P1]], killed [[P5]]
; CHECK:    [[RES2:%[0-9]+]]:ppr = AND_PPzPP [[P2]], [[P2]], killed [[P6]]
; CHECK:    [[RES3:%[0-9]+]]:ppr = AND_PPzPP [[P3]], [[P3]], killed [[P7]]
; CHECK:    $p0 = COPY [[RES0]]
; CHECK:    $p1 = COPY [[RES1]]
; CHECK:    $p2 = COPY [[RES2]]
; CHECK:    $p3 = COPY [[RES3]]
; CHECK:    RET_ReallyLR implicit $p0, implicit $p1, implicit $p2, implicit $p3
  %p0 = extractvalue [2 x <vscale x 16 x i1>] %arg1, 0
  %p1 = extractvalue [2 x <vscale x 16 x i1>] %arg1, 1
  %p2 = extractvalue [2 x <vscale x 16 x i1>] %arg3, 0
  %p3 = extractvalue [2 x <vscale x 16 x i1>] %arg3, 1
  %p4 = extractvalue [4 x <vscale x 16 x i1>] %arg2, 0
  %p5 = extractvalue [4 x <vscale x 16 x i1>] %arg2, 1
  %p6 = extractvalue [4 x <vscale x 16 x i1>] %arg2, 2
  %p7 = extractvalue [4 x <vscale x 16 x i1>] %arg2, 3
  %r0 = and <vscale x 16 x i1> %p0, %p4
  %r1 = and <vscale x 16 x i1> %p1, %p5
  %r2 = and <vscale x 16 x i1> %p2, %p6
  %r3 = and <vscale x 16 x i1> %p3, %p7
  %1 = insertvalue  [4 x <vscale x 16 x i1>] poison, <vscale x 16 x i1> %r0, 0
  %2 = insertvalue  [4 x <vscale x 16 x i1>] %1, <vscale x 16 x i1> %r1, 1
  %3 = insertvalue  [4 x <vscale x 16 x i1>] %2, <vscale x 16 x i1> %r2, 2
  %4 = insertvalue  [4 x <vscale x 16 x i1>] %3, <vscale x 16 x i1> %r3, 3
  ret [4 x <vscale x 16 x i1>] %4
}

; Test that z8 and z9, passed by reference, are loaded from a location that is passed on the stack.
; i.e.     x0 =   %x0
;             :
;          x7 =   %x7
;          z0 =   %z0
;             :
;          z7 =   %z7
;        [sp] =  &%z8
;      [sp+8] =  &%z9
;
define aarch64_sve_vector_pcs <vscale x 4 x i32> @callee_with_many_gpr_sve_arg(i64 %x0, i64 %x1, i64 %x2, i64 %x3, i64 %x4, i64 %x5, i64 %x6, i64 %x7, <vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z3, <vscale x 4 x i32> %z4, <vscale x 4 x i32> %z5, <vscale x 4 x i32> %z6, <vscale x 4 x i32> %z7, <vscale x 2 x i64> %z8, <vscale x 4 x i32> %z9) {
; CHECK: name: callee_with_many_gpr_sve_arg
; CHECK: fixedStack:
; CHECK:      - { id: 0, type: default, offset: 8, size: 8, alignment: 8, stack-id: default,
; CHECK-DAG: [[BASE:%[0-9]+]]:gpr64common = LDRXui %fixed-stack.0, 0
; CHECK-DAG: [[RES:%[0-9]+]]:zpr = LDR_ZXI killed [[BASE]]
; CHECK-DAG: $z0 = COPY [[RES]]
; CHECK: RET_ReallyLR implicit $z0
  ret <vscale x 4 x i32> %z9
}

; Test that z8 and z9 are passed by reference, where reference is passed on the stack.
define aarch64_sve_vector_pcs <vscale x 4 x i32> @caller_with_many_gpr_sve_arg(i64 %x, <vscale x 4 x i32> %z, <vscale x 2 x i64> %z2) {
; CHECK: name: caller_with_many_gpr_sve_arg
; CHECK: stack:
; CHECK:      - { id: 0, name: '', type: default, offset: 0, size: 16, alignment: 16,
; CHECK-NEXT:     stack-id: scalable-vector
; CHECK:      - { id: 1, name: '', type: default, offset: 0, size: 16, alignment: 16,
; CHECK-NEXT:     stack-id: scalable-vector
; CHECK-DAG: STR_ZXI %{{[0-9]+}}, %stack.0, 0
; CHECK-DAG: STR_ZXI %{{[0-9]+}}, %stack.1, 0
; CHECK-DAG: [[BASE1:%[0-9]+]]:gpr64common = ADDXri %stack.0, 0
; CHECK-DAG: [[BASE2:%[0-9]+]]:gpr64common = ADDXri %stack.1, 0
; CHECK-DAG: [[SP:%[0-9]+]]:gpr64sp = COPY $sp
; CHECK-DAG: STRXui killed [[BASE1]], [[SP]], 0
; CHECK-DAG: STRXui killed [[BASE2]], [[SP]], 1
; CHECK:     BL @callee_with_many_gpr_sve_arg
; CHECK:     RET_ReallyLR implicit $z0
  %ret = call aarch64_sve_vector_pcs <vscale x 4 x i32> @callee_with_many_gpr_sve_arg(i64 %x, i64 %x, i64 %x, i64 %x, i64 %x, i64 %x, i64 %x, i64 %x, <vscale x 4 x i32> %z, <vscale x 4 x i32> %z, <vscale x 4 x i32> %z, <vscale x 4 x i32> %z, <vscale x 4 x i32> %z, <vscale x 4 x i32> %z, <vscale x 4 x i32> %z, <vscale x 4 x i32> %z, <vscale x 2 x i64> %z2, <vscale x 4 x i32> %z)
  ret <vscale x 4 x i32> %ret
}
