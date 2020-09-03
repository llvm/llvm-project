; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -stop-after=finalize-isel < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

; Test that z8 and z9, passed in by reference, are correctly loaded from x0 and x1.
; i.e. z0 =  %z0
;         :
;      z7 =  %z7
;      x0 = &%z8
;      x1 = &%z9
define aarch64_sve_vector_pcs <vscale x 4 x i32> @callee_with_many_sve_arg(<vscale x 4 x i32> %z0, <vscale x 4 x i32> %z1, <vscale x 4 x i32> %z2, <vscale x 4 x i32> %z3, <vscale x 4 x i32> %z4, <vscale x 4 x i32> %z5, <vscale x 4 x i32> %z6, <vscale x 4 x i32> %z7, <vscale x 4 x i32> %z8, <vscale x 4 x i32> %z9) {
; CHECK: name: callee_with_many_sve_arg
; CHECK-DAG: [[BASE:%[0-9]+]]:gpr64common = COPY $x1
; CHECK-DAG: [[PTRUE:%[0-9]+]]:ppr_3b = PTRUE_S 31
; CHECK-DAG: [[RES:%[0-9]+]]:zpr = LD1W_IMM killed [[PTRUE]], [[BASE]]
; CHECK-DAG: $z0 = COPY [[RES]]
; CHECK:     RET_ReallyLR implicit $z0
  ret <vscale x 4 x i32> %z9
}

; Test that z8 and z9 are passed by reference.
define aarch64_sve_vector_pcs <vscale x 4 x i32> @caller_with_many_sve_arg(<vscale x 4 x i32> %z) {
; CHECK: name: caller_with_many_sve_arg
; CHECK: stack:
; CHECK:      - { id: 0, name: '', type: default, offset: 0, size: 16, alignment: 16,
; CHECK-NEXT:     stack-id: sve-vec
; CHECK:      - { id: 1, name: '', type: default, offset: 0, size: 16, alignment: 16,
; CHECK-NEXT:     stack-id: sve-vec
; CHECK-DAG:  [[PTRUE:%[0-9]+]]:ppr_3b = PTRUE_S 31
; CHECK-DAG:  ST1W_IMM %{{[0-9]+}}, [[PTRUE]], %stack.1, 0
; CHECK-DAG:  ST1W_IMM %{{[0-9]+}}, [[PTRUE]], %stack.0, 0
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
define aarch64_sve_vector_pcs <vscale x 4 x i1> @callee_with_many_svepred_arg(<vscale x 4 x i1> %p0, <vscale x 4 x i1> %p1, <vscale x 4 x i1> %p2, <vscale x 4 x i1> %p3, <vscale x 4 x i1> %p4, <vscale x 4 x i1> %p5) {
; CHECK: name: callee_with_many_svepred_arg
; CHECK-DAG: [[BASE:%[0-9]+]]:gpr64common = COPY $x1
; CHECK-DAG: [[RES:%[0-9]+]]:ppr = LDR_PXI [[BASE]], 0
; CHECK-DAG: $p0 = COPY [[RES]]
; CHECK:     RET_ReallyLR implicit $p0
  ret <vscale x 4 x i1> %p5
}

; Test that p4 and p5 are passed by reference.
define aarch64_sve_vector_pcs <vscale x 4 x i1> @caller_with_many_svepred_arg(<vscale x 4 x i1> %p) {
; CHECK: name: caller_with_many_svepred_arg
; CHECK: stack:
; CHECK:      - { id: 0, name: '', type: default, offset: 0, size: 1, alignment: 4,
; CHECK-NEXT:     stack-id: sve-vec
; CHECK:      - { id: 1, name: '', type: default, offset: 0, size: 1, alignment: 4,
; CHECK-NEXT:     stack-id: sve-vec
; CHECK-DAG: STR_PXI %{{[0-9]+}}, %stack.0, 0
; CHECK-DAG: STR_PXI %{{[0-9]+}}, %stack.1, 0
; CHECK-DAG: [[BASE1:%[0-9]+]]:gpr64sp = ADDXri %stack.0, 0
; CHECK-DAG: [[BASE2:%[0-9]+]]:gpr64sp = ADDXri %stack.1, 0
; CHECK-DAG: $x0 = COPY [[BASE1]]
; CHECK-DAG: $x1 = COPY [[BASE2]]
; CHECK-NEXT: BL @callee_with_many_svepred_arg
; CHECK:     RET_ReallyLR implicit $p0
  %ret = call aarch64_sve_vector_pcs <vscale x 4 x i1> @callee_with_many_svepred_arg(<vscale x 4 x i1> %p, <vscale x 4 x i1> %p, <vscale x 4 x i1> %p, <vscale x 4 x i1> %p, <vscale x 4 x i1> %p, <vscale x 4 x i1> %p)
  ret <vscale x 4 x i1> %ret
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
; CHECK-DAG: [[PTRUE:%[0-9]+]]:ppr_3b = PTRUE_S 31
; CHECK-DAG: [[RES:%[0-9]+]]:zpr = LD1W_IMM killed [[PTRUE]], killed [[BASE]]
; CHECK-DAG: $z0 = COPY [[RES]]
; CHECK: RET_ReallyLR implicit $z0
  ret <vscale x 4 x i32> %z9
}

; Test that z8 and z9 are passed by reference, where reference is passed on the stack.
define aarch64_sve_vector_pcs <vscale x 4 x i32> @caller_with_many_gpr_sve_arg(i64 %x, <vscale x 4 x i32> %z, <vscale x 2 x i64> %z2) {
; CHECK: name: caller_with_many_gpr_sve_arg
; CHECK: stack:
; CHECK:      - { id: 0, name: '', type: default, offset: 0, size: 16, alignment: 16,
; CHECK-NEXT:     stack-id: sve-vec
; CHECK:      - { id: 1, name: '', type: default, offset: 0, size: 16, alignment: 16,
; CHECK-NEXT:     stack-id: sve-vec
; CHECK-DAG: [[PTRUE_S:%[0-9]+]]:ppr_3b = PTRUE_S 31
; CHECK-DAG: [[PTRUE_D:%[0-9]+]]:ppr_3b = PTRUE_D 31
; CHECK-DAG: ST1D_IMM %{{[0-9]+}}, killed [[PTRUE_D]], %stack.0, 0
; CHECK-DAG: ST1W_IMM %{{[0-9]+}}, killed [[PTRUE_S]], %stack.1, 0
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
