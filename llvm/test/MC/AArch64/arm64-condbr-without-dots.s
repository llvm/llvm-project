// RUN: llvm-mc -triple arm64-apple-ios -o - %s | FileCheck %s
        
        beq lbl
        bne lbl
        bcs lbl
        bhs lbl
        blo lbl
        bcc lbl
        bmi lbl
        bpl lbl
        bvs lbl
        bvc lbl
        bhi lbl
        bls lbl
        bge lbl
        blt lbl
        bgt lbl
        ble lbl
        bal lbl

// CHECK: b.eq lbl
// CHECK: b.ne lbl
// CHECK: b.hs lbl
// CHECK: b.hs lbl
// CHECK: b.lo lbl
// CHECK: b.lo lbl
// CHECK: b.mi lbl
// CHECK: b.pl lbl
// CHECK: b.vs lbl
// CHECK: b.vc lbl
// CHECK: b.hi lbl
// CHECK: b.ls lbl
// CHECK: b.ge lbl
// CHECK: b.lt lbl
// CHECK: b.gt lbl
// CHECK: b.le lbl
// CHECK: b.al lbl
