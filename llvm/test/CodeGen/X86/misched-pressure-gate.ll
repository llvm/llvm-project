; Check that GenericScheduler skips the RegExcess/RegCritical heuristics only in
; regions where they cannot pay: large, already over the register pressure
; limit, and throughput rather than latency bound.
;
; RUN: llc < %s -mtriple=x86_64-- -mcpu=x86-64 -debug-only=machine-scheduler \
; RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=GATED
;
; The gate is off when the option is zero.
; RUN: llc < %s -mtriple=x86_64-- -mcpu=x86-64 -debug-only=machine-scheduler \
; RUN:   -misched-pressure-gate-windows=0 -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NOGATE
;
; The absolute floor alone is enough to hold it off ...
; RUN: llc < %s -mtriple=x86_64-- -mcpu=x86-64 -debug-only=machine-scheduler \
; RUN:   -misched-pressure-gate-min-instrs=100000 -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NOGATE
;
; ... and so is the out-of-order window ratio alone.
; RUN: llc < %s -mtriple=x86_64-- -mcpu=x86-64 -debug-only=machine-scheduler \
; RUN:   -misched-pressure-gate-windows=100 -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NOGATE
;
; REQUIRES: asserts

; The loop below keeps many <16 x i32> values live at once, so register
; pressure is far above the limit whatever order the scheduler picks, and the
; work is independent, so throughput and not dependence height sets the pace.

; When the gate fires, the region is announced and neither pressure heuristic
; may pick a node in it again.  Checking the picks as well as the message
; matters: the message alone would still be printed if the guards in
; tryCandidate() were dropped and the analysis left in place.

; GATED: Pressure heuristics disabled
; GATED-NOT: Pick {{Top|Bot}} REG-

; With the gate held off, the region is not announced and RegExcess does pick.

; NOGATE-NOT: Pressure heuristics disabled
; NOGATE: Pick {{Top|Bot}} REG-EXCESS

define void @pressure_loop(ptr %p, ptr %q, i64 %n) {
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %a0 = getelementptr inbounds <16 x i32>, ptr %p, i64 0
  %v0 = load <16 x i32>, ptr %a0
  %a1 = getelementptr inbounds <16 x i32>, ptr %p, i64 1
  %v1 = load <16 x i32>, ptr %a1
  %a2 = getelementptr inbounds <16 x i32>, ptr %p, i64 2
  %v2 = load <16 x i32>, ptr %a2
  %a3 = getelementptr inbounds <16 x i32>, ptr %p, i64 3
  %v3 = load <16 x i32>, ptr %a3
  %a4 = getelementptr inbounds <16 x i32>, ptr %p, i64 4
  %v4 = load <16 x i32>, ptr %a4
  %a5 = getelementptr inbounds <16 x i32>, ptr %p, i64 5
  %v5 = load <16 x i32>, ptr %a5
  %a6 = getelementptr inbounds <16 x i32>, ptr %p, i64 6
  %v6 = load <16 x i32>, ptr %a6
  %a7 = getelementptr inbounds <16 x i32>, ptr %p, i64 7
  %v7 = load <16 x i32>, ptr %a7
  %a8 = getelementptr inbounds <16 x i32>, ptr %p, i64 8
  %v8 = load <16 x i32>, ptr %a8
  %a9 = getelementptr inbounds <16 x i32>, ptr %p, i64 9
  %v9 = load <16 x i32>, ptr %a9
  %a10 = getelementptr inbounds <16 x i32>, ptr %p, i64 10
  %v10 = load <16 x i32>, ptr %a10
  %a11 = getelementptr inbounds <16 x i32>, ptr %p, i64 11
  %v11 = load <16 x i32>, ptr %a11
  %a12 = getelementptr inbounds <16 x i32>, ptr %p, i64 12
  %v12 = load <16 x i32>, ptr %a12
  %a13 = getelementptr inbounds <16 x i32>, ptr %p, i64 13
  %v13 = load <16 x i32>, ptr %a13
  %a14 = getelementptr inbounds <16 x i32>, ptr %p, i64 14
  %v14 = load <16 x i32>, ptr %a14
  %a15 = getelementptr inbounds <16 x i32>, ptr %p, i64 15
  %v15 = load <16 x i32>, ptr %a15
  %a16 = getelementptr inbounds <16 x i32>, ptr %p, i64 16
  %v16 = load <16 x i32>, ptr %a16
  %a17 = getelementptr inbounds <16 x i32>, ptr %p, i64 17
  %v17 = load <16 x i32>, ptr %a17
  %a18 = getelementptr inbounds <16 x i32>, ptr %p, i64 18
  %v18 = load <16 x i32>, ptr %a18
  %a19 = getelementptr inbounds <16 x i32>, ptr %p, i64 19
  %v19 = load <16 x i32>, ptr %a19
  %a20 = getelementptr inbounds <16 x i32>, ptr %p, i64 20
  %v20 = load <16 x i32>, ptr %a20
  %a21 = getelementptr inbounds <16 x i32>, ptr %p, i64 21
  %v21 = load <16 x i32>, ptr %a21
  %a22 = getelementptr inbounds <16 x i32>, ptr %p, i64 22
  %v22 = load <16 x i32>, ptr %a22
  %a23 = getelementptr inbounds <16 x i32>, ptr %p, i64 23
  %v23 = load <16 x i32>, ptr %a23
  %a24 = getelementptr inbounds <16 x i32>, ptr %p, i64 24
  %v24 = load <16 x i32>, ptr %a24
  %a25 = getelementptr inbounds <16 x i32>, ptr %p, i64 25
  %v25 = load <16 x i32>, ptr %a25
  %a26 = getelementptr inbounds <16 x i32>, ptr %p, i64 26
  %v26 = load <16 x i32>, ptr %a26
  %a27 = getelementptr inbounds <16 x i32>, ptr %p, i64 27
  %v27 = load <16 x i32>, ptr %a27
  %a28 = getelementptr inbounds <16 x i32>, ptr %p, i64 28
  %v28 = load <16 x i32>, ptr %a28
  %a29 = getelementptr inbounds <16 x i32>, ptr %p, i64 29
  %v29 = load <16 x i32>, ptr %a29
  %a30 = getelementptr inbounds <16 x i32>, ptr %p, i64 30
  %v30 = load <16 x i32>, ptr %a30
  %a31 = getelementptr inbounds <16 x i32>, ptr %p, i64 31
  %v31 = load <16 x i32>, ptr %a31
  %a32 = getelementptr inbounds <16 x i32>, ptr %p, i64 32
  %v32 = load <16 x i32>, ptr %a32
  %a33 = getelementptr inbounds <16 x i32>, ptr %p, i64 33
  %v33 = load <16 x i32>, ptr %a33
  %a34 = getelementptr inbounds <16 x i32>, ptr %p, i64 34
  %v34 = load <16 x i32>, ptr %a34
  %a35 = getelementptr inbounds <16 x i32>, ptr %p, i64 35
  %v35 = load <16 x i32>, ptr %a35
  %a36 = getelementptr inbounds <16 x i32>, ptr %p, i64 36
  %v36 = load <16 x i32>, ptr %a36
  %a37 = getelementptr inbounds <16 x i32>, ptr %p, i64 37
  %v37 = load <16 x i32>, ptr %a37
  %a38 = getelementptr inbounds <16 x i32>, ptr %p, i64 38
  %v38 = load <16 x i32>, ptr %a38
  %a39 = getelementptr inbounds <16 x i32>, ptr %p, i64 39
  %v39 = load <16 x i32>, ptr %a39
  %a40 = getelementptr inbounds <16 x i32>, ptr %p, i64 40
  %v40 = load <16 x i32>, ptr %a40
  %a41 = getelementptr inbounds <16 x i32>, ptr %p, i64 41
  %v41 = load <16 x i32>, ptr %a41
  %a42 = getelementptr inbounds <16 x i32>, ptr %p, i64 42
  %v42 = load <16 x i32>, ptr %a42
  %a43 = getelementptr inbounds <16 x i32>, ptr %p, i64 43
  %v43 = load <16 x i32>, ptr %a43
  %a44 = getelementptr inbounds <16 x i32>, ptr %p, i64 44
  %v44 = load <16 x i32>, ptr %a44
  %m0 = add <16 x i32> %v0, %v1
  %m1 = add <16 x i32> %v1, %v2
  %m2 = add <16 x i32> %v2, %v3
  %m3 = add <16 x i32> %v3, %v4
  %m4 = add <16 x i32> %v4, %v5
  %m5 = add <16 x i32> %v5, %v6
  %m6 = add <16 x i32> %v6, %v7
  %m7 = add <16 x i32> %v7, %v8
  %m8 = add <16 x i32> %v8, %v9
  %m9 = add <16 x i32> %v9, %v10
  %m10 = add <16 x i32> %v10, %v11
  %m11 = add <16 x i32> %v11, %v12
  %m12 = add <16 x i32> %v12, %v13
  %m13 = add <16 x i32> %v13, %v14
  %m14 = add <16 x i32> %v14, %v15
  %m15 = add <16 x i32> %v15, %v16
  %m16 = add <16 x i32> %v16, %v17
  %m17 = add <16 x i32> %v17, %v18
  %m18 = add <16 x i32> %v18, %v19
  %m19 = add <16 x i32> %v19, %v20
  %m20 = add <16 x i32> %v20, %v21
  %m21 = add <16 x i32> %v21, %v22
  %m22 = add <16 x i32> %v22, %v23
  %m23 = add <16 x i32> %v23, %v24
  %m24 = add <16 x i32> %v24, %v25
  %m25 = add <16 x i32> %v25, %v26
  %m26 = add <16 x i32> %v26, %v27
  %m27 = add <16 x i32> %v27, %v28
  %m28 = add <16 x i32> %v28, %v29
  %m29 = add <16 x i32> %v29, %v30
  %m30 = add <16 x i32> %v30, %v31
  %m31 = add <16 x i32> %v31, %v32
  %m32 = add <16 x i32> %v32, %v33
  %m33 = add <16 x i32> %v33, %v34
  %m34 = add <16 x i32> %v34, %v35
  %m35 = add <16 x i32> %v35, %v36
  %m36 = add <16 x i32> %v36, %v37
  %m37 = add <16 x i32> %v37, %v38
  %m38 = add <16 x i32> %v38, %v39
  %m39 = add <16 x i32> %v39, %v40
  %m40 = add <16 x i32> %v40, %v41
  %m41 = add <16 x i32> %v41, %v42
  %m42 = add <16 x i32> %v42, %v43
  %m43 = add <16 x i32> %v43, %v44
  %m44 = add <16 x i32> %v44, %v0
  %x0 = xor <16 x i32> %m0, %v2
  %x1 = xor <16 x i32> %m1, %v3
  %x2 = xor <16 x i32> %m2, %v4
  %x3 = xor <16 x i32> %m3, %v5
  %x4 = xor <16 x i32> %m4, %v6
  %x5 = xor <16 x i32> %m5, %v7
  %x6 = xor <16 x i32> %m6, %v8
  %x7 = xor <16 x i32> %m7, %v9
  %x8 = xor <16 x i32> %m8, %v10
  %x9 = xor <16 x i32> %m9, %v11
  %x10 = xor <16 x i32> %m10, %v12
  %x11 = xor <16 x i32> %m11, %v13
  %x12 = xor <16 x i32> %m12, %v14
  %x13 = xor <16 x i32> %m13, %v15
  %x14 = xor <16 x i32> %m14, %v16
  %x15 = xor <16 x i32> %m15, %v17
  %x16 = xor <16 x i32> %m16, %v18
  %x17 = xor <16 x i32> %m17, %v19
  %x18 = xor <16 x i32> %m18, %v20
  %x19 = xor <16 x i32> %m19, %v21
  %x20 = xor <16 x i32> %m20, %v22
  %x21 = xor <16 x i32> %m21, %v23
  %x22 = xor <16 x i32> %m22, %v24
  %x23 = xor <16 x i32> %m23, %v25
  %x24 = xor <16 x i32> %m24, %v26
  %x25 = xor <16 x i32> %m25, %v27
  %x26 = xor <16 x i32> %m26, %v28
  %x27 = xor <16 x i32> %m27, %v29
  %x28 = xor <16 x i32> %m28, %v30
  %x29 = xor <16 x i32> %m29, %v31
  %x30 = xor <16 x i32> %m30, %v32
  %x31 = xor <16 x i32> %m31, %v33
  %x32 = xor <16 x i32> %m32, %v34
  %x33 = xor <16 x i32> %m33, %v35
  %x34 = xor <16 x i32> %m34, %v36
  %x35 = xor <16 x i32> %m35, %v37
  %x36 = xor <16 x i32> %m36, %v38
  %x37 = xor <16 x i32> %m37, %v39
  %x38 = xor <16 x i32> %m38, %v40
  %x39 = xor <16 x i32> %m39, %v41
  %x40 = xor <16 x i32> %m40, %v42
  %x41 = xor <16 x i32> %m41, %v43
  %x42 = xor <16 x i32> %m42, %v44
  %x43 = xor <16 x i32> %m43, %v0
  %x44 = xor <16 x i32> %m44, %v1
  %b0 = getelementptr inbounds <16 x i32>, ptr %q, i64 0
  store <16 x i32> %x0, ptr %b0
  %b1 = getelementptr inbounds <16 x i32>, ptr %q, i64 1
  store <16 x i32> %x1, ptr %b1
  %b2 = getelementptr inbounds <16 x i32>, ptr %q, i64 2
  store <16 x i32> %x2, ptr %b2
  %b3 = getelementptr inbounds <16 x i32>, ptr %q, i64 3
  store <16 x i32> %x3, ptr %b3
  %b4 = getelementptr inbounds <16 x i32>, ptr %q, i64 4
  store <16 x i32> %x4, ptr %b4
  %b5 = getelementptr inbounds <16 x i32>, ptr %q, i64 5
  store <16 x i32> %x5, ptr %b5
  %b6 = getelementptr inbounds <16 x i32>, ptr %q, i64 6
  store <16 x i32> %x6, ptr %b6
  %b7 = getelementptr inbounds <16 x i32>, ptr %q, i64 7
  store <16 x i32> %x7, ptr %b7
  %b8 = getelementptr inbounds <16 x i32>, ptr %q, i64 8
  store <16 x i32> %x8, ptr %b8
  %b9 = getelementptr inbounds <16 x i32>, ptr %q, i64 9
  store <16 x i32> %x9, ptr %b9
  %b10 = getelementptr inbounds <16 x i32>, ptr %q, i64 10
  store <16 x i32> %x10, ptr %b10
  %b11 = getelementptr inbounds <16 x i32>, ptr %q, i64 11
  store <16 x i32> %x11, ptr %b11
  %b12 = getelementptr inbounds <16 x i32>, ptr %q, i64 12
  store <16 x i32> %x12, ptr %b12
  %b13 = getelementptr inbounds <16 x i32>, ptr %q, i64 13
  store <16 x i32> %x13, ptr %b13
  %b14 = getelementptr inbounds <16 x i32>, ptr %q, i64 14
  store <16 x i32> %x14, ptr %b14
  %b15 = getelementptr inbounds <16 x i32>, ptr %q, i64 15
  store <16 x i32> %x15, ptr %b15
  %b16 = getelementptr inbounds <16 x i32>, ptr %q, i64 16
  store <16 x i32> %x16, ptr %b16
  %b17 = getelementptr inbounds <16 x i32>, ptr %q, i64 17
  store <16 x i32> %x17, ptr %b17
  %b18 = getelementptr inbounds <16 x i32>, ptr %q, i64 18
  store <16 x i32> %x18, ptr %b18
  %b19 = getelementptr inbounds <16 x i32>, ptr %q, i64 19
  store <16 x i32> %x19, ptr %b19
  %b20 = getelementptr inbounds <16 x i32>, ptr %q, i64 20
  store <16 x i32> %x20, ptr %b20
  %b21 = getelementptr inbounds <16 x i32>, ptr %q, i64 21
  store <16 x i32> %x21, ptr %b21
  %b22 = getelementptr inbounds <16 x i32>, ptr %q, i64 22
  store <16 x i32> %x22, ptr %b22
  %b23 = getelementptr inbounds <16 x i32>, ptr %q, i64 23
  store <16 x i32> %x23, ptr %b23
  %b24 = getelementptr inbounds <16 x i32>, ptr %q, i64 24
  store <16 x i32> %x24, ptr %b24
  %b25 = getelementptr inbounds <16 x i32>, ptr %q, i64 25
  store <16 x i32> %x25, ptr %b25
  %b26 = getelementptr inbounds <16 x i32>, ptr %q, i64 26
  store <16 x i32> %x26, ptr %b26
  %b27 = getelementptr inbounds <16 x i32>, ptr %q, i64 27
  store <16 x i32> %x27, ptr %b27
  %b28 = getelementptr inbounds <16 x i32>, ptr %q, i64 28
  store <16 x i32> %x28, ptr %b28
  %b29 = getelementptr inbounds <16 x i32>, ptr %q, i64 29
  store <16 x i32> %x29, ptr %b29
  %b30 = getelementptr inbounds <16 x i32>, ptr %q, i64 30
  store <16 x i32> %x30, ptr %b30
  %b31 = getelementptr inbounds <16 x i32>, ptr %q, i64 31
  store <16 x i32> %x31, ptr %b31
  %b32 = getelementptr inbounds <16 x i32>, ptr %q, i64 32
  store <16 x i32> %x32, ptr %b32
  %b33 = getelementptr inbounds <16 x i32>, ptr %q, i64 33
  store <16 x i32> %x33, ptr %b33
  %b34 = getelementptr inbounds <16 x i32>, ptr %q, i64 34
  store <16 x i32> %x34, ptr %b34
  %b35 = getelementptr inbounds <16 x i32>, ptr %q, i64 35
  store <16 x i32> %x35, ptr %b35
  %b36 = getelementptr inbounds <16 x i32>, ptr %q, i64 36
  store <16 x i32> %x36, ptr %b36
  %b37 = getelementptr inbounds <16 x i32>, ptr %q, i64 37
  store <16 x i32> %x37, ptr %b37
  %b38 = getelementptr inbounds <16 x i32>, ptr %q, i64 38
  store <16 x i32> %x38, ptr %b38
  %b39 = getelementptr inbounds <16 x i32>, ptr %q, i64 39
  store <16 x i32> %x39, ptr %b39
  %b40 = getelementptr inbounds <16 x i32>, ptr %q, i64 40
  store <16 x i32> %x40, ptr %b40
  %b41 = getelementptr inbounds <16 x i32>, ptr %q, i64 41
  store <16 x i32> %x41, ptr %b41
  %b42 = getelementptr inbounds <16 x i32>, ptr %q, i64 42
  store <16 x i32> %x42, ptr %b42
  %b43 = getelementptr inbounds <16 x i32>, ptr %q, i64 43
  store <16 x i32> %x43, ptr %b43
  %b44 = getelementptr inbounds <16 x i32>, ptr %q, i64 44
  store <16 x i32> %x44, ptr %b44
  %iv.next = add i64 %iv, 1
  %c = icmp slt i64 %iv.next, %n
  br i1 %c, label %loop, label %exit

exit:
  ret void
}
