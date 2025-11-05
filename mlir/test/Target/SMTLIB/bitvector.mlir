// RUN: mlir-translate --export-smtlib %s | FileCheck %s
// RUN: mlir-translate --export-smtlib --smtlibexport-inline-single-use-values %s | FileCheck %s --check-prefix=CHECK-INLINED

smt.solver () : () -> () {
  %c0_bv32 = smt.bv.constant #smt.bv<0> : !smt.bv<32>

  // CHECK: (assert (let (([[V0:.+]] (bvneg #x00000000)))
  // CHECK:         (let (([[V1:.+]] (= [[V0]] #x00000000)))
  // CHECK:         [[V1]])))

  // CHECK-INLINED: (assert (= (bvneg #x00000000) #x00000000))
  %0 = smt.bv.neg %c0_bv32 : !smt.bv<32>
  %a0 = smt.eq %0, %c0_bv32 : !smt.bv<32>
  smt.assert %a0

  // CHECK: (assert (let (([[V2:.+]] (bvadd #x00000000 #x00000000)))
  // CHECK:         (let (([[V3:.+]] (= [[V2]] #x00000000)))
  // CHECK:         [[V3]])))

  // CHECK-INLINED: (assert (= (bvadd #x00000000 #x00000000) #x00000000))
  %1 = smt.bv.add %c0_bv32, %c0_bv32 : !smt.bv<32>
  %a1 = smt.eq %1, %c0_bv32 : !smt.bv<32>
  smt.assert %a1

  // CHECK: (assert (let (([[V4:.+]] (bvmul #x00000000 #x00000000)))
  // CHECK:         (let (([[V5:.+]] (= [[V4]] #x00000000)))
  // CHECK:         [[V5]])))

  // CHECK-INLINED: (assert (= (bvmul #x00000000 #x00000000) #x00000000))
  %3 = smt.bv.mul %c0_bv32, %c0_bv32 : !smt.bv<32>
  %a3 = smt.eq %3, %c0_bv32 : !smt.bv<32>
  smt.assert %a3

  // CHECK: (assert (let (([[V6:.+]] (bvurem #x00000000 #x00000000)))
  // CHECK:         (let (([[V7:.+]] (= [[V6]] #x00000000)))
  // CHECK:         [[V7]])))

  // CHECK-INLINED: (assert (= (bvurem #x00000000 #x00000000) #x00000000))
  %4 = smt.bv.urem %c0_bv32, %c0_bv32 : !smt.bv<32>
  %a4 = smt.eq %4, %c0_bv32 : !smt.bv<32>
  smt.assert %a4

  // CHECK: (assert (let (([[V8:.+]] (bvsrem #x00000000 #x00000000)))
  // CHECK:         (let (([[V9:.+]] (= [[V8]] #x00000000)))
  // CHECK:         [[V9]])))

  // CHECK-INLINED: (assert (= (bvsrem #x00000000 #x00000000) #x00000000))
  %5 = smt.bv.srem %c0_bv32, %c0_bv32 : !smt.bv<32>
  %a5 = smt.eq %5, %c0_bv32 : !smt.bv<32>
  smt.assert %a5

  // CHECK: (assert (let (([[V10:.+]] (bvsmod #x00000000 #x00000000)))
  // CHECK:         (let (([[V11:.+]] (= [[V10]] #x00000000)))
  // CHECK:         [[V11]])))

  // CHECK-INLINED: (assert (= (bvsmod #x00000000 #x00000000) #x00000000))
  %7 = smt.bv.smod %c0_bv32, %c0_bv32 : !smt.bv<32>
  %a7 = smt.eq %7, %c0_bv32 : !smt.bv<32>
  smt.assert %a7

  // CHECK: (assert (let (([[V12:.+]] (bvshl #x00000000 #x00000000)))
  // CHECK:         (let (([[V13:.+]] (= [[V12]] #x00000000)))
  // CHECK:         [[V13]])))

  // CHECK-INLINED: (assert (= (bvshl #x00000000 #x00000000) #x00000000))
  %8 = smt.bv.shl %c0_bv32, %c0_bv32 : !smt.bv<32>
  %a8 = smt.eq %8, %c0_bv32 : !smt.bv<32>
  smt.assert %a8

  // CHECK: (assert (let (([[V14:.+]] (bvlshr #x00000000 #x00000000)))
  // CHECK:         (let (([[V15:.+]] (= [[V14]] #x00000000)))
  // CHECK:         [[V15]])))

  // CHECK-INLINED: (assert (= (bvlshr #x00000000 #x00000000) #x00000000))
  %9 = smt.bv.lshr %c0_bv32, %c0_bv32 : !smt.bv<32>
  %a9 = smt.eq %9, %c0_bv32 : !smt.bv<32>
  smt.assert %a9

  // CHECK: (assert (let (([[V16:.+]] (bvashr #x00000000 #x00000000)))
  // CHECK:         (let (([[V17:.+]] (= [[V16]] #x00000000)))
  // CHECK:         [[V17]])))

  // CHECK-INLINED: (assert (= (bvashr #x00000000 #x00000000) #x00000000))
  %10 = smt.bv.ashr %c0_bv32, %c0_bv32 : !smt.bv<32>
  %a10 = smt.eq %10, %c0_bv32 : !smt.bv<32>
  smt.assert %a10

  // CHECK: (assert (let (([[V18:.+]] (bvudiv #x00000000 #x00000000)))
  // CHECK:         (let (([[V19:.+]] (= [[V18]] #x00000000)))
  // CHECK:         [[V19]])))

  // CHECK-INLINED: (assert (= (bvudiv #x00000000 #x00000000) #x00000000))
  %11 = smt.bv.udiv %c0_bv32, %c0_bv32 : !smt.bv<32>
  %a11 = smt.eq %11, %c0_bv32 : !smt.bv<32>
  smt.assert %a11

  // CHECK: (assert (let (([[V20:.+]] (bvsdiv #x00000000 #x00000000)))
  // CHECK:         (let (([[V21:.+]] (= [[V20]] #x00000000)))
  // CHECK:         [[V21]])))

  // CHECK-INLINED: (assert (= (bvsdiv #x00000000 #x00000000) #x00000000))
  %12 = smt.bv.sdiv %c0_bv32, %c0_bv32 : !smt.bv<32>
  %a12 = smt.eq %12, %c0_bv32 : !smt.bv<32>
  smt.assert %a12

  // CHECK: (assert (let (([[V22:.+]] (bvnot #x00000000)))
  // CHECK:         (let (([[V23:.+]] (= [[V22]] #x00000000)))
  // CHECK:         [[V23]])))

  // CHECK-INLINED: (assert (= (bvnot #x00000000) #x00000000))
  %13 = smt.bv.not %c0_bv32 : !smt.bv<32>
  %a13 = smt.eq %13, %c0_bv32 : !smt.bv<32>
  smt.assert %a13

  // CHECK: (assert (let (([[V24:.+]] (bvand #x00000000 #x00000000)))
  // CHECK:         (let (([[V25:.+]] (= [[V24]] #x00000000)))
  // CHECK:         [[V25]])))

  // CHECK-INLINED: (assert (= (bvand #x00000000 #x00000000) #x00000000))
  %14 = smt.bv.and %c0_bv32, %c0_bv32 : !smt.bv<32>
  %a14 = smt.eq %14, %c0_bv32 : !smt.bv<32>
  smt.assert %a14

  // CHECK: (assert (let (([[V26:.+]] (bvor #x00000000 #x00000000)))
  // CHECK:         (let (([[V27:.+]] (= [[V26]] #x00000000)))
  // CHECK:         [[V27]])))

  // CHECK-INLINED: (assert (= (bvor #x00000000 #x00000000) #x00000000))
  %15 = smt.bv.or %c0_bv32, %c0_bv32 : !smt.bv<32>
  %a15 = smt.eq %15, %c0_bv32 : !smt.bv<32>
  smt.assert %a15

  // CHECK: (assert (let (([[V28:.+]] (bvxor #x00000000 #x00000000)))
  // CHECK:         (let (([[V29:.+]] (= [[V28]] #x00000000)))
  // CHECK:         [[V29]])))

  // CHECK-INLINED: (assert (= (bvxor #x00000000 #x00000000) #x00000000))
  %16 = smt.bv.xor %c0_bv32, %c0_bv32 : !smt.bv<32>
  %a16 = smt.eq %16, %c0_bv32 : !smt.bv<32>
  smt.assert %a16

  // CHECK: (assert (let (([[V30:.+]] (bvslt #x00000000 #x00000000)))
  // CHECK:         [[V30]]))

  // CHECK-INLINED: (assert (bvslt #x00000000 #x00000000))
  %27 = smt.bv.cmp slt %c0_bv32, %c0_bv32 : !smt.bv<32>
  smt.assert %27

  // CHECK: (assert (let (([[V31:.+]] (bvsle #x00000000 #x00000000)))
  // CHECK:         [[V31]]))

  // CHECK-INLINED: (assert (bvsle #x00000000 #x00000000))
  %28 = smt.bv.cmp sle %c0_bv32, %c0_bv32 : !smt.bv<32>
  smt.assert %28

  // CHECK: (assert (let (([[V32:.+]] (bvsgt #x00000000 #x00000000)))
  // CHECK:         [[V32]]))

  // CHECK-INLINED: (assert (bvsgt #x00000000 #x00000000))
  %29 = smt.bv.cmp sgt %c0_bv32, %c0_bv32 : !smt.bv<32>
  smt.assert %29

  // CHECK: (assert (let (([[V33:.+]] (bvsge #x00000000 #x00000000)))
  // CHECK:         [[V33]]))

  // CHECK-INLINED: (assert (bvsge #x00000000 #x00000000))
  %30 = smt.bv.cmp sge %c0_bv32, %c0_bv32 : !smt.bv<32>
  smt.assert %30

  // CHECK: (assert (let (([[V34:.+]] (bvult #x00000000 #x00000000)))
  // CHECK:         [[V34]]))

  // CHECK-INLINED: (assert (bvult #x00000000 #x00000000))
  %31 = smt.bv.cmp ult %c0_bv32, %c0_bv32 : !smt.bv<32>
  smt.assert %31

  // CHECK: (assert (let (([[V35:.+]] (bvule #x00000000 #x00000000)))
  // CHECK:         [[V35]]))

  // CHECK-INLINED: (assert (bvule #x00000000 #x00000000))
  %32 = smt.bv.cmp ule %c0_bv32, %c0_bv32 : !smt.bv<32>
  smt.assert %32

  // CHECK: (assert (let (([[V36:.+]] (bvugt #x00000000 #x00000000)))
  // CHECK:         [[V36]]))

  // CHECK-INLINED: (assert (bvugt #x00000000 #x00000000))
  %33 = smt.bv.cmp ugt %c0_bv32, %c0_bv32 : !smt.bv<32>
  smt.assert %33

  // CHECK: (assert (let (([[V37:.+]] (bvuge #x00000000 #x00000000)))
  // CHECK:         [[V37]]))

  // CHECK-INLINED: (assert (bvuge #x00000000 #x00000000))
  %34 = smt.bv.cmp uge %c0_bv32, %c0_bv32 : !smt.bv<32>
  smt.assert %34

  // CHECK: (assert (let (([[V38:.+]] (concat #x00000000 #x00000000)))
  // CHECK:         (let (([[V39:.+]] ((_ extract 23 8) [[V38]])))
  // CHECK:         (let (([[V40:.+]] ((_ repeat 2) [[V39]])))
  // CHECK:         (let (([[V41:.+]] (= [[V40]] #x00000000)))
  // CHECK:         [[V41]])))))

  // CHECK-INLINED: (assert (= ((_ repeat 2) ((_ extract 23 8) (concat #x00000000 #x00000000))) #x00000000))
  %35 = smt.bv.concat %c0_bv32, %c0_bv32 : !smt.bv<32>, !smt.bv<32>
  %36 = smt.bv.extract %35 from 8 : (!smt.bv<64>) -> !smt.bv<16>
  %37 = smt.bv.repeat 2 times %36 : !smt.bv<16>
  %a37 = smt.eq %37, %c0_bv32 : !smt.bv<32>
  smt.assert %a37

  // CHECK: (reset)
  // CHECK-INLINED: (reset)
}
