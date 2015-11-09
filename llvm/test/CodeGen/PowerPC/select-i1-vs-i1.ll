; RUN: llc < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; FIXME: We should check the operands to the cr* logical operation itself, but
; unfortunately, FileCheck does not yet understand how to do arithmetic, so we
; can't do so without introducing a register-allocation dependency.

define signext i32 @testi32slt(i32 signext %c1, i32 signext %c2, i32 signext %c3, i32 signext %c4, i32 signext %a1, i32 signext %a2) #0 {
entry:
  %cmp1 = icmp eq i32 %c3, %c4
  %cmp3tmp = icmp eq i32 %c1, %c2
  %cmp3 = icmp slt i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, i32 %a1, i32 %a2
  ret i32 %cond

; CHECK-LABEL: @testi32slt
; CHECK-DAG: cmpw {{[0-9]+}}, 5, 6
; CHECK-DAG: cmpw {{[0-9]+}}, 3, 4
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: isel 3, 7, 8, [[REG1]]
; CHECK: blr
}

define signext i32 @testi32ult(i32 signext %c1, i32 signext %c2, i32 signext %c3, i32 signext %c4, i32 signext %a1, i32 signext %a2) #0 {
entry:
  %cmp1 = icmp eq i32 %c3, %c4
  %cmp3tmp = icmp eq i32 %c1, %c2
  %cmp3 = icmp ult i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, i32 %a1, i32 %a2
  ret i32 %cond

; CHECK-LABEL: @testi32ult
; CHECK-DAG: cmpw {{[0-9]+}}, 5, 6
; CHECK-DAG: cmpw {{[0-9]+}}, 3, 4
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: isel 3, 7, 8, [[REG1]]
; CHECK: blr
}

define signext i32 @testi32sle(i32 signext %c1, i32 signext %c2, i32 signext %c3, i32 signext %c4, i32 signext %a1, i32 signext %a2) #0 {
entry:
  %cmp1 = icmp eq i32 %c3, %c4
  %cmp3tmp = icmp eq i32 %c1, %c2
  %cmp3 = icmp sle i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, i32 %a1, i32 %a2
  ret i32 %cond

; CHECK-LABEL: @testi32sle
; CHECK-DAG: cmpw {{[0-9]+}}, 5, 6
; CHECK-DAG: cmpw {{[0-9]+}}, 3, 4
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: isel 3, 7, 8, [[REG1]]
; CHECK: blr
}

define signext i32 @testi32ule(i32 signext %c1, i32 signext %c2, i32 signext %c3, i32 signext %c4, i32 signext %a1, i32 signext %a2) #0 {
entry:
  %cmp1 = icmp eq i32 %c3, %c4
  %cmp3tmp = icmp eq i32 %c1, %c2
  %cmp3 = icmp ule i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, i32 %a1, i32 %a2
  ret i32 %cond

; CHECK-LABEL: @testi32ule
; CHECK-DAG: cmpw {{[0-9]+}}, 5, 6
; CHECK-DAG: cmpw {{[0-9]+}}, 3, 4
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: isel 3, 7, 8, [[REG1]]
; CHECK: blr
}

define signext i32 @testi32eq(i32 signext %c1, i32 signext %c2, i32 signext %c3, i32 signext %c4, i32 signext %a1, i32 signext %a2) #0 {
entry:
  %cmp1 = icmp eq i32 %c3, %c4
  %cmp3tmp = icmp eq i32 %c1, %c2
  %cmp3 = icmp eq i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, i32 %a1, i32 %a2
  ret i32 %cond

; CHECK-LABEL: @testi32eq
; CHECK-DAG: cmpw {{[0-9]+}}, 5, 6
; CHECK-DAG: cmpw {{[0-9]+}}, 3, 4
; CHECK: creqv [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: isel 3, 7, 8, [[REG1]]
; CHECK: blr
}

define signext i32 @testi32sge(i32 signext %c1, i32 signext %c2, i32 signext %c3, i32 signext %c4, i32 signext %a1, i32 signext %a2) #0 {
entry:
  %cmp1 = icmp eq i32 %c3, %c4
  %cmp3tmp = icmp eq i32 %c1, %c2
  %cmp3 = icmp sge i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, i32 %a1, i32 %a2
  ret i32 %cond

; CHECK-LABEL: @testi32sge
; CHECK-DAG: cmpw {{[0-9]+}}, 5, 6
; CHECK-DAG: cmpw {{[0-9]+}}, 3, 4
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: isel 3, 7, 8, [[REG1]]
; CHECK: blr
}

define signext i32 @testi32uge(i32 signext %c1, i32 signext %c2, i32 signext %c3, i32 signext %c4, i32 signext %a1, i32 signext %a2) #0 {
entry:
  %cmp1 = icmp eq i32 %c3, %c4
  %cmp3tmp = icmp eq i32 %c1, %c2
  %cmp3 = icmp uge i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, i32 %a1, i32 %a2
  ret i32 %cond

; CHECK-LABEL: @testi32uge
; CHECK-DAG: cmpw {{[0-9]+}}, 5, 6
; CHECK-DAG: cmpw {{[0-9]+}}, 3, 4
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: isel 3, 7, 8, [[REG1]]
; CHECK: blr
}

define signext i32 @testi32sgt(i32 signext %c1, i32 signext %c2, i32 signext %c3, i32 signext %c4, i32 signext %a1, i32 signext %a2) #0 {
entry:
  %cmp1 = icmp eq i32 %c3, %c4
  %cmp3tmp = icmp eq i32 %c1, %c2
  %cmp3 = icmp sgt i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, i32 %a1, i32 %a2
  ret i32 %cond

; CHECK-LABEL: @testi32sgt
; CHECK-DAG: cmpw {{[0-9]+}}, 5, 6
; CHECK-DAG: cmpw {{[0-9]+}}, 3, 4
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: isel 3, 7, 8, [[REG1]]
; CHECK: blr
}

define signext i32 @testi32ugt(i32 signext %c1, i32 signext %c2, i32 signext %c3, i32 signext %c4, i32 signext %a1, i32 signext %a2) #0 {
entry:
  %cmp1 = icmp eq i32 %c3, %c4
  %cmp3tmp = icmp eq i32 %c1, %c2
  %cmp3 = icmp ugt i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, i32 %a1, i32 %a2
  ret i32 %cond

; CHECK-LABEL: @testi32ugt
; CHECK-DAG: cmpw {{[0-9]+}}, 5, 6
; CHECK-DAG: cmpw {{[0-9]+}}, 3, 4
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: isel 3, 7, 8, [[REG1]]
; CHECK: blr
}

define signext i32 @testi32ne(i32 signext %c1, i32 signext %c2, i32 signext %c3, i32 signext %c4, i32 signext %a1, i32 signext %a2) #0 {
entry:
  %cmp1 = icmp eq i32 %c3, %c4
  %cmp3tmp = icmp eq i32 %c1, %c2
  %cmp3 = icmp ne i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, i32 %a1, i32 %a2
  ret i32 %cond

; CHECK-LABEL: @testi32ne
; CHECK-DAG: cmpw {{[0-9]+}}, 5, 6
; CHECK-DAG: cmpw {{[0-9]+}}, 3, 4
; CHECK: crxor [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: isel 3, 7, 8, [[REG1]]
; CHECK: blr
}

define i64 @testi64slt(i64 %c1, i64 %c2, i64 %c3, i64 %c4, i64 %a1, i64 %a2) #0 {
entry:
  %cmp1 = icmp eq i64 %c3, %c4
  %cmp3tmp = icmp eq i64 %c1, %c2
  %cmp3 = icmp slt i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, i64 %a1, i64 %a2
  ret i64 %cond

; CHECK-LABEL: @testi64slt
; CHECK-DAG: cmpd {{([0-9]+, )?}}5, 6
; CHECK-DAG: cmpd {{([0-9]+, )?}}3, 4
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: isel 3, 7, 8, [[REG1]]
; CHECK: blr
}

define i64 @testi64ult(i64 %c1, i64 %c2, i64 %c3, i64 %c4, i64 %a1, i64 %a2) #0 {
entry:
  %cmp1 = icmp eq i64 %c3, %c4
  %cmp3tmp = icmp eq i64 %c1, %c2
  %cmp3 = icmp ult i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, i64 %a1, i64 %a2
  ret i64 %cond

; CHECK-LABEL: @testi64ult
; CHECK-DAG: cmpd {{([0-9]+, )?}}5, 6
; CHECK-DAG: cmpd {{([0-9]+, )?}}3, 4
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: isel 3, 7, 8, [[REG1]]
; CHECK: blr
}

define i64 @testi64sle(i64 %c1, i64 %c2, i64 %c3, i64 %c4, i64 %a1, i64 %a2) #0 {
entry:
  %cmp1 = icmp eq i64 %c3, %c4
  %cmp3tmp = icmp eq i64 %c1, %c2
  %cmp3 = icmp sle i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, i64 %a1, i64 %a2
  ret i64 %cond

; CHECK-LABEL: @testi64sle
; CHECK-DAG: cmpd {{([0-9]+, )?}}5, 6
; CHECK-DAG: cmpd {{([0-9]+, )?}}3, 4
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: isel 3, 7, 8, [[REG1]]
; CHECK: blr
}

define i64 @testi64ule(i64 %c1, i64 %c2, i64 %c3, i64 %c4, i64 %a1, i64 %a2) #0 {
entry:
  %cmp1 = icmp eq i64 %c3, %c4
  %cmp3tmp = icmp eq i64 %c1, %c2
  %cmp3 = icmp ule i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, i64 %a1, i64 %a2
  ret i64 %cond

; CHECK-LABEL: @testi64ule
; CHECK-DAG: cmpd {{([0-9]+, )?}}5, 6
; CHECK-DAG: cmpd {{([0-9]+, )?}}3, 4
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: isel 3, 7, 8, [[REG1]]
; CHECK: blr
}

define i64 @testi64eq(i64 %c1, i64 %c2, i64 %c3, i64 %c4, i64 %a1, i64 %a2) #0 {
entry:
  %cmp1 = icmp eq i64 %c3, %c4
  %cmp3tmp = icmp eq i64 %c1, %c2
  %cmp3 = icmp eq i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, i64 %a1, i64 %a2
  ret i64 %cond

; CHECK-LABEL: @testi64eq
; CHECK-DAG: cmpd {{([0-9]+, )?}}5, 6
; CHECK-DAG: cmpd {{([0-9]+, )?}}3, 4
; CHECK: creqv [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: isel 3, 7, 8, [[REG1]]
; CHECK: blr
}

define i64 @testi64sge(i64 %c1, i64 %c2, i64 %c3, i64 %c4, i64 %a1, i64 %a2) #0 {
entry:
  %cmp1 = icmp eq i64 %c3, %c4
  %cmp3tmp = icmp eq i64 %c1, %c2
  %cmp3 = icmp sge i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, i64 %a1, i64 %a2
  ret i64 %cond

; CHECK-LABEL: @testi64sge
; CHECK-DAG: cmpd {{([0-9]+, )?}}5, 6
; CHECK-DAG: cmpd {{([0-9]+, )?}}3, 4
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: isel 3, 7, 8, [[REG1]]
; CHECK: blr
}

define i64 @testi64uge(i64 %c1, i64 %c2, i64 %c3, i64 %c4, i64 %a1, i64 %a2) #0 {
entry:
  %cmp1 = icmp eq i64 %c3, %c4
  %cmp3tmp = icmp eq i64 %c1, %c2
  %cmp3 = icmp uge i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, i64 %a1, i64 %a2
  ret i64 %cond

; CHECK-LABEL: @testi64uge
; CHECK-DAG: cmpd {{([0-9]+, )?}}5, 6
; CHECK-DAG: cmpd {{([0-9]+, )?}}3, 4
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: isel 3, 7, 8, [[REG1]]
; CHECK: blr
}

define i64 @testi64sgt(i64 %c1, i64 %c2, i64 %c3, i64 %c4, i64 %a1, i64 %a2) #0 {
entry:
  %cmp1 = icmp eq i64 %c3, %c4
  %cmp3tmp = icmp eq i64 %c1, %c2
  %cmp3 = icmp sgt i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, i64 %a1, i64 %a2
  ret i64 %cond

; CHECK-LABEL: @testi64sgt
; CHECK-DAG: cmpd {{([0-9]+, )?}}5, 6
; CHECK-DAG: cmpd {{([0-9]+, )?}}3, 4
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: isel 3, 7, 8, [[REG1]]
; CHECK: blr
}

define i64 @testi64ugt(i64 %c1, i64 %c2, i64 %c3, i64 %c4, i64 %a1, i64 %a2) #0 {
entry:
  %cmp1 = icmp eq i64 %c3, %c4
  %cmp3tmp = icmp eq i64 %c1, %c2
  %cmp3 = icmp ugt i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, i64 %a1, i64 %a2
  ret i64 %cond

; CHECK-LABEL: @testi64ugt
; CHECK-DAG: cmpd {{([0-9]+, )?}}5, 6
; CHECK-DAG: cmpd {{([0-9]+, )?}}3, 4
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: isel 3, 7, 8, [[REG1]]
; CHECK: blr
}

define i64 @testi64ne(i64 %c1, i64 %c2, i64 %c3, i64 %c4, i64 %a1, i64 %a2) #0 {
entry:
  %cmp1 = icmp eq i64 %c3, %c4
  %cmp3tmp = icmp eq i64 %c1, %c2
  %cmp3 = icmp ne i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, i64 %a1, i64 %a2
  ret i64 %cond

; CHECK-LABEL: @testi64ne
; CHECK-DAG: cmpd {{([0-9]+, )?}}5, 6
; CHECK-DAG: cmpd {{([0-9]+, )?}}3, 4
; CHECK: crxor [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: isel 3, 7, 8, [[REG1]]
; CHECK: blr
}

define float @testfloatslt(float %c1, float %c2, float %c3, float %c4, float %a1, float %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp slt i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, float %a1, float %a2
  ret float %cond

; CHECK-LABEL: @testfloatslt
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: fmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: fmr 1, 5
; CHECK: blr
}

define float @testfloatult(float %c1, float %c2, float %c3, float %c4, float %a1, float %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp ult i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, float %a1, float %a2
  ret float %cond

; CHECK-LABEL: @testfloatult
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: fmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: fmr 1, 5
; CHECK: blr
}

define float @testfloatsle(float %c1, float %c2, float %c3, float %c4, float %a1, float %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp sle i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, float %a1, float %a2
  ret float %cond

; CHECK-LABEL: @testfloatsle
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: fmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: fmr 1, 5
; CHECK: blr
}

define float @testfloatule(float %c1, float %c2, float %c3, float %c4, float %a1, float %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp ule i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, float %a1, float %a2
  ret float %cond

; CHECK-LABEL: @testfloatule
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: fmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: fmr 1, 5
; CHECK: blr
}

define float @testfloateq(float %c1, float %c2, float %c3, float %c4, float %a1, float %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp eq i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, float %a1, float %a2
  ret float %cond

; CHECK-LABEL: @testfloateq
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: creqv [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: fmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: fmr 1, 5
; CHECK: blr
}

define float @testfloatsge(float %c1, float %c2, float %c3, float %c4, float %a1, float %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp sge i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, float %a1, float %a2
  ret float %cond

; CHECK-LABEL: @testfloatsge
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: fmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: fmr 1, 5
; CHECK: blr
}

define float @testfloatuge(float %c1, float %c2, float %c3, float %c4, float %a1, float %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp uge i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, float %a1, float %a2
  ret float %cond

; CHECK-LABEL: @testfloatuge
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: fmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: fmr 1, 5
; CHECK: blr
}

define float @testfloatsgt(float %c1, float %c2, float %c3, float %c4, float %a1, float %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp sgt i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, float %a1, float %a2
  ret float %cond

; CHECK-LABEL: @testfloatsgt
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: fmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: fmr 1, 5
; CHECK: blr
}

define float @testfloatugt(float %c1, float %c2, float %c3, float %c4, float %a1, float %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp ugt i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, float %a1, float %a2
  ret float %cond

; CHECK-LABEL: @testfloatugt
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: fmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: fmr 1, 5
; CHECK: blr
}

define float @testfloatne(float %c1, float %c2, float %c3, float %c4, float %a1, float %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp ne i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, float %a1, float %a2
  ret float %cond

; CHECK-LABEL: @testfloatne
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crxor [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: fmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: fmr 1, 5
; CHECK: blr
}

define double @testdoubleslt(double %c1, double %c2, double %c3, double %c4, double %a1, double %a2) #0 {
entry:
  %cmp1 = fcmp oeq double %c3, %c4
  %cmp3tmp = fcmp oeq double %c1, %c2
  %cmp3 = icmp slt i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, double %a1, double %a2
  ret double %cond

; CHECK-LABEL: @testdoubleslt
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: fmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: fmr 1, 5
; CHECK: blr
}

define double @testdoubleult(double %c1, double %c2, double %c3, double %c4, double %a1, double %a2) #0 {
entry:
  %cmp1 = fcmp oeq double %c3, %c4
  %cmp3tmp = fcmp oeq double %c1, %c2
  %cmp3 = icmp ult i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, double %a1, double %a2
  ret double %cond

; CHECK-LABEL: @testdoubleult
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: fmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: fmr 1, 5
; CHECK: blr
}

define double @testdoublesle(double %c1, double %c2, double %c3, double %c4, double %a1, double %a2) #0 {
entry:
  %cmp1 = fcmp oeq double %c3, %c4
  %cmp3tmp = fcmp oeq double %c1, %c2
  %cmp3 = icmp sle i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, double %a1, double %a2
  ret double %cond

; CHECK-LABEL: @testdoublesle
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: fmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: fmr 1, 5
; CHECK: blr
}

define double @testdoubleule(double %c1, double %c2, double %c3, double %c4, double %a1, double %a2) #0 {
entry:
  %cmp1 = fcmp oeq double %c3, %c4
  %cmp3tmp = fcmp oeq double %c1, %c2
  %cmp3 = icmp ule i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, double %a1, double %a2
  ret double %cond

; CHECK-LABEL: @testdoubleule
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: fmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: fmr 1, 5
; CHECK: blr
}

define double @testdoubleeq(double %c1, double %c2, double %c3, double %c4, double %a1, double %a2) #0 {
entry:
  %cmp1 = fcmp oeq double %c3, %c4
  %cmp3tmp = fcmp oeq double %c1, %c2
  %cmp3 = icmp eq i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, double %a1, double %a2
  ret double %cond

; CHECK-LABEL: @testdoubleeq
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: creqv [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: fmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: fmr 1, 5
; CHECK: blr
}

define double @testdoublesge(double %c1, double %c2, double %c3, double %c4, double %a1, double %a2) #0 {
entry:
  %cmp1 = fcmp oeq double %c3, %c4
  %cmp3tmp = fcmp oeq double %c1, %c2
  %cmp3 = icmp sge i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, double %a1, double %a2
  ret double %cond

; CHECK-LABEL: @testdoublesge
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: fmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: fmr 1, 5
; CHECK: blr
}

define double @testdoubleuge(double %c1, double %c2, double %c3, double %c4, double %a1, double %a2) #0 {
entry:
  %cmp1 = fcmp oeq double %c3, %c4
  %cmp3tmp = fcmp oeq double %c1, %c2
  %cmp3 = icmp uge i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, double %a1, double %a2
  ret double %cond

; CHECK-LABEL: @testdoubleuge
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: fmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: fmr 1, 5
; CHECK: blr
}

define double @testdoublesgt(double %c1, double %c2, double %c3, double %c4, double %a1, double %a2) #0 {
entry:
  %cmp1 = fcmp oeq double %c3, %c4
  %cmp3tmp = fcmp oeq double %c1, %c2
  %cmp3 = icmp sgt i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, double %a1, double %a2
  ret double %cond

; CHECK-LABEL: @testdoublesgt
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: fmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: fmr 1, 5
; CHECK: blr
}

define double @testdoubleugt(double %c1, double %c2, double %c3, double %c4, double %a1, double %a2) #0 {
entry:
  %cmp1 = fcmp oeq double %c3, %c4
  %cmp3tmp = fcmp oeq double %c1, %c2
  %cmp3 = icmp ugt i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, double %a1, double %a2
  ret double %cond

; CHECK-LABEL: @testdoubleugt
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: fmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: fmr 1, 5
; CHECK: blr
}

define double @testdoublene(double %c1, double %c2, double %c3, double %c4, double %a1, double %a2) #0 {
entry:
  %cmp1 = fcmp oeq double %c3, %c4
  %cmp3tmp = fcmp oeq double %c1, %c2
  %cmp3 = icmp ne i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, double %a1, double %a2
  ret double %cond

; CHECK-LABEL: @testdoublene
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crxor [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: fmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: fmr 1, 5
; CHECK: blr
}

define <4 x float> @testv4floatslt(float %c1, float %c2, float %c3, float %c4, <4 x float> %a1, <4 x float> %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp slt i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x float> %a1, <4 x float> %a2
  ret <4 x float> %cond

; FIXME: This test (and the other v4f32 tests) should use the same bclr
; technique as the v2f64 tests below.

; CHECK-LABEL: @testv4floatslt
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK-DAG: xxlor [[REG2:[0-9]+]], 34, 34
; CHECK-DAG: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: xxlor [[REG2]], 35, 35
; CHECK: .LBB[[BB]]:
; CHECK: xxlor 34, [[REG2]], [[REG2]]
; CHECK: blr
}

define <4 x float> @testv4floatult(float %c1, float %c2, float %c3, float %c4, <4 x float> %a1, <4 x float> %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp ult i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x float> %a1, <4 x float> %a2
  ret <4 x float> %cond

; CHECK-LABEL: @testv4floatult
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK-DAG: xxlor [[REG2:[0-9]+]], 34, 34
; CHECK-DAG: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: xxlor [[REG2]], 35, 35
; CHECK: .LBB[[BB]]:
; CHECK: xxlor 34, [[REG2]], [[REG2]]
; CHECK: blr
}

define <4 x float> @testv4floatsle(float %c1, float %c2, float %c3, float %c4, <4 x float> %a1, <4 x float> %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp sle i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x float> %a1, <4 x float> %a2
  ret <4 x float> %cond

; CHECK-LABEL: @testv4floatsle
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK-DAG: xxlor [[REG2:[0-9]+]], 34, 34
; CHECK-DAG: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: xxlor [[REG2]], 35, 35
; CHECK: .LBB[[BB]]:
; CHECK: xxlor 34, [[REG2]], [[REG2]]
; CHECK: blr
}

define <4 x float> @testv4floatule(float %c1, float %c2, float %c3, float %c4, <4 x float> %a1, <4 x float> %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp ule i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x float> %a1, <4 x float> %a2
  ret <4 x float> %cond

; CHECK-LABEL: @testv4floatule
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK-DAG: xxlor [[REG2:[0-9]+]], 34, 34
; CHECK-DAG: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: xxlor [[REG2]], 35, 35
; CHECK: .LBB[[BB]]:
; CHECK: xxlor 34, [[REG2]], [[REG2]]
; CHECK: blr
}

define <4 x float> @testv4floateq(float %c1, float %c2, float %c3, float %c4, <4 x float> %a1, <4 x float> %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp eq i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x float> %a1, <4 x float> %a2
  ret <4 x float> %cond

; CHECK-LABEL: @testv4floateq
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK-DAG: xxlor [[REG2:[0-9]+]], 34, 34
; CHECK-DAG: creqv [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: xxlor [[REG2]], 35, 35
; CHECK: .LBB[[BB]]:
; CHECK: xxlor 34, [[REG2]], [[REG2]]
; CHECK: blr
}

define <4 x float> @testv4floatsge(float %c1, float %c2, float %c3, float %c4, <4 x float> %a1, <4 x float> %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp sge i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x float> %a1, <4 x float> %a2
  ret <4 x float> %cond

; CHECK-LABEL: @testv4floatsge
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK-DAG: xxlor [[REG2:[0-9]+]], 34, 34
; CHECK-DAG: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: xxlor [[REG2]], 35, 35
; CHECK: .LBB[[BB]]:
; CHECK: xxlor 34, [[REG2]], [[REG2]]
; CHECK: blr
}

define <4 x float> @testv4floatuge(float %c1, float %c2, float %c3, float %c4, <4 x float> %a1, <4 x float> %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp uge i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x float> %a1, <4 x float> %a2
  ret <4 x float> %cond

; CHECK-LABEL: @testv4floatuge
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK-DAG: xxlor [[REG2:[0-9]+]], 34, 34
; CHECK-DAG: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: xxlor [[REG2]], 35, 35
; CHECK: .LBB[[BB]]:
; CHECK: xxlor 34, [[REG2]], [[REG2]]
; CHECK: blr
}

define <4 x float> @testv4floatsgt(float %c1, float %c2, float %c3, float %c4, <4 x float> %a1, <4 x float> %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp sgt i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x float> %a1, <4 x float> %a2
  ret <4 x float> %cond

; CHECK-LABEL: @testv4floatsgt
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK-DAG: xxlor [[REG2:[0-9]+]], 34, 34
; CHECK-DAG: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: xxlor [[REG2]], 35, 35
; CHECK: .LBB[[BB]]:
; CHECK: xxlor 34, [[REG2]], [[REG2]]
; CHECK: blr
}

define <4 x float> @testv4floatugt(float %c1, float %c2, float %c3, float %c4, <4 x float> %a1, <4 x float> %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp ugt i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x float> %a1, <4 x float> %a2
  ret <4 x float> %cond

; CHECK-LABEL: @testv4floatugt
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK-DAG: xxlor [[REG2:[0-9]+]], 34, 34
; CHECK-DAG: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: xxlor [[REG2]], 35, 35
; CHECK: .LBB[[BB]]:
; CHECK: xxlor 34, [[REG2]], [[REG2]]
; CHECK: blr
}

define <4 x float> @testv4floatne(float %c1, float %c2, float %c3, float %c4, <4 x float> %a1, <4 x float> %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp ne i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x float> %a1, <4 x float> %a2
  ret <4 x float> %cond

; CHECK-LABEL: @testv4floatne
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK-DAG: xxlor [[REG2:[0-9]+]], 34, 34
; CHECK-DAG: crxor [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: xxlor [[REG2]], 35, 35
; CHECK: .LBB[[BB]]:
; CHECK: xxlor 34, [[REG2]], [[REG2]]
; CHECK: blr
}

define ppc_fp128 @testppc_fp128eq(ppc_fp128 %c1, ppc_fp128 %c2, ppc_fp128 %c3, ppc_fp128 %c4, ppc_fp128 %a1, ppc_fp128 %a2) #0 {
entry:
  %cmp1 = fcmp oeq ppc_fp128 %c3, %c4
  %cmp3tmp = fcmp oeq ppc_fp128 %c1, %c2
  %cmp3 = icmp eq i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, ppc_fp128 %a1, ppc_fp128 %a2
  ret ppc_fp128 %cond

; FIXME: Because of the way that the late SELECT_* pseudo-instruction expansion
; works, we end up with two blocks with the same predicate. These could be
; combined.

; CHECK-LABEL: @testppc_fp128eq
; CHECK-DAG: fcmpu {{[0-9]+}}, 6, 8
; CHECK-DAG: fcmpu {{[0-9]+}}, 5, 7
; CHECK-DAG: fcmpu {{[0-9]+}}, 2, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 3
; CHECK: crand [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: crand [[REG2:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: creqv [[REG3:[0-9]+]], [[REG2]], [[REG1]]
; CHECK: bc 12, [[REG3]], .LBB[[BB1:[0-9_]+]]
; CHECK: fmr 9, 11
; CHECK: .LBB[[BB1]]:
; CHECK: bc 12, [[REG3]], .LBB[[BB2:[0-9_]+]]
; CHECK: fmr 10, 12
; CHECK: .LBB[[BB2]]:
; CHECK-DAG: fmr 1, 9
; CHECK-DAG: fmr 2, 10
; CHECK: blr
}

define <2 x double> @testv2doubleslt(float %c1, float %c2, float %c3, float %c4, <2 x double> %a1, <2 x double> %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp slt i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <2 x double> %a1, <2 x double> %a2
  ret <2 x double> %cond

; CHECK-LABEL: @testv2doubleslt
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bclr 12, [[REG1]], 0
; CHECK: vor 2, 3, 3
; CHECK: blr
}

define <2 x double> @testv2doubleult(float %c1, float %c2, float %c3, float %c4, <2 x double> %a1, <2 x double> %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp ult i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <2 x double> %a1, <2 x double> %a2
  ret <2 x double> %cond

; CHECK-LABEL: @testv2doubleult
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bclr 12, [[REG1]], 0
; CHECK: vor 2, 3, 3
; CHECK: blr
}

define <2 x double> @testv2doublesle(float %c1, float %c2, float %c3, float %c4, <2 x double> %a1, <2 x double> %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp sle i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <2 x double> %a1, <2 x double> %a2
  ret <2 x double> %cond

; CHECK-LABEL: @testv2doublesle
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bclr 12, [[REG1]], 0
; CHECK: vor 2, 3, 3
; CHECK: blr
}

define <2 x double> @testv2doubleule(float %c1, float %c2, float %c3, float %c4, <2 x double> %a1, <2 x double> %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp ule i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <2 x double> %a1, <2 x double> %a2
  ret <2 x double> %cond

; CHECK-LABEL: @testv2doubleule
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bclr 12, [[REG1]], 0
; CHECK: vor 2, 3, 3
; CHECK: blr
}

define <2 x double> @testv2doubleeq(float %c1, float %c2, float %c3, float %c4, <2 x double> %a1, <2 x double> %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp eq i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <2 x double> %a1, <2 x double> %a2
  ret <2 x double> %cond

; CHECK-LABEL: @testv2doubleeq
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: creqv [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bclr 12, [[REG1]], 0
; CHECK: vor 2, 3, 3
; CHECK: blr
}

define <2 x double> @testv2doublesge(float %c1, float %c2, float %c3, float %c4, <2 x double> %a1, <2 x double> %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp sge i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <2 x double> %a1, <2 x double> %a2
  ret <2 x double> %cond

; CHECK-LABEL: @testv2doublesge
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bclr 12, [[REG1]], 0
; CHECK: vor 2, 3, 3
; CHECK: blr
}

define <2 x double> @testv2doubleuge(float %c1, float %c2, float %c3, float %c4, <2 x double> %a1, <2 x double> %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp uge i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <2 x double> %a1, <2 x double> %a2
  ret <2 x double> %cond

; CHECK-LABEL: @testv2doubleuge
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bclr 12, [[REG1]], 0
; CHECK: vor 2, 3, 3
; CHECK: blr
}

define <2 x double> @testv2doublesgt(float %c1, float %c2, float %c3, float %c4, <2 x double> %a1, <2 x double> %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp sgt i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <2 x double> %a1, <2 x double> %a2
  ret <2 x double> %cond

; CHECK-LABEL: @testv2doublesgt
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bclr 12, [[REG1]], 0
; CHECK: vor 2, 3, 3
; CHECK: blr
}

define <2 x double> @testv2doubleugt(float %c1, float %c2, float %c3, float %c4, <2 x double> %a1, <2 x double> %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp ugt i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <2 x double> %a1, <2 x double> %a2
  ret <2 x double> %cond

; CHECK-LABEL: @testv2doubleugt
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bclr 12, [[REG1]], 0
; CHECK: vor 2, 3, 3
; CHECK: blr
}

define <2 x double> @testv2doublene(float %c1, float %c2, float %c3, float %c4, <2 x double> %a1, <2 x double> %a2) #0 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp ne i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <2 x double> %a1, <2 x double> %a2
  ret <2 x double> %cond

; CHECK-LABEL: @testv2doublene
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crxor [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bclr 12, [[REG1]], 0
; CHECK: vor 2, 3, 3
; CHECK: blr
}

define <4 x double> @testqv4doubleslt(float %c1, float %c2, float %c3, float %c4, <4 x double> %a1, <4 x double> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp slt i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x double> %a1, <4 x double> %a2
  ret <4 x double> %cond

; CHECK-LABEL: @testqv4doubleslt
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

define <4 x double> @testqv4doubleult(float %c1, float %c2, float %c3, float %c4, <4 x double> %a1, <4 x double> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp ult i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x double> %a1, <4 x double> %a2
  ret <4 x double> %cond

; CHECK-LABEL: @testqv4doubleult
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

define <4 x double> @testqv4doublesle(float %c1, float %c2, float %c3, float %c4, <4 x double> %a1, <4 x double> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp sle i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x double> %a1, <4 x double> %a2
  ret <4 x double> %cond

; CHECK-LABEL: @testqv4doublesle
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

define <4 x double> @testqv4doubleule(float %c1, float %c2, float %c3, float %c4, <4 x double> %a1, <4 x double> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp ule i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x double> %a1, <4 x double> %a2
  ret <4 x double> %cond

; CHECK-LABEL: @testqv4doubleule
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

define <4 x double> @testqv4doubleeq(float %c1, float %c2, float %c3, float %c4, <4 x double> %a1, <4 x double> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp eq i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x double> %a1, <4 x double> %a2
  ret <4 x double> %cond

; CHECK-LABEL: @testqv4doubleeq
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: creqv [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

define <4 x double> @testqv4doublesge(float %c1, float %c2, float %c3, float %c4, <4 x double> %a1, <4 x double> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp sge i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x double> %a1, <4 x double> %a2
  ret <4 x double> %cond

; CHECK-LABEL: @testqv4doublesge
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

define <4 x double> @testqv4doubleuge(float %c1, float %c2, float %c3, float %c4, <4 x double> %a1, <4 x double> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp uge i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x double> %a1, <4 x double> %a2
  ret <4 x double> %cond

; CHECK-LABEL: @testqv4doubleuge
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

define <4 x double> @testqv4doublesgt(float %c1, float %c2, float %c3, float %c4, <4 x double> %a1, <4 x double> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp sgt i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x double> %a1, <4 x double> %a2
  ret <4 x double> %cond

; CHECK-LABEL: @testqv4doublesgt
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

define <4 x double> @testqv4doubleugt(float %c1, float %c2, float %c3, float %c4, <4 x double> %a1, <4 x double> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp ugt i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x double> %a1, <4 x double> %a2
  ret <4 x double> %cond

; CHECK-LABEL: @testqv4doubleugt
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

define <4 x double> @testqv4doublene(float %c1, float %c2, float %c3, float %c4, <4 x double> %a1, <4 x double> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp ne i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x double> %a1, <4 x double> %a2
  ret <4 x double> %cond

; CHECK-LABEL: @testqv4doublene
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crxor [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

define <4 x float> @testqv4floatslt(float %c1, float %c2, float %c3, float %c4, <4 x float> %a1, <4 x float> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp slt i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x float> %a1, <4 x float> %a2
  ret <4 x float> %cond

; CHECK-LABEL: @testqv4floatslt
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

define <4 x float> @testqv4floatult(float %c1, float %c2, float %c3, float %c4, <4 x float> %a1, <4 x float> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp ult i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x float> %a1, <4 x float> %a2
  ret <4 x float> %cond

; CHECK-LABEL: @testqv4floatult
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

define <4 x float> @testqv4floatsle(float %c1, float %c2, float %c3, float %c4, <4 x float> %a1, <4 x float> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp sle i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x float> %a1, <4 x float> %a2
  ret <4 x float> %cond

; CHECK-LABEL: @testqv4floatsle
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

define <4 x float> @testqv4floatule(float %c1, float %c2, float %c3, float %c4, <4 x float> %a1, <4 x float> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp ule i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x float> %a1, <4 x float> %a2
  ret <4 x float> %cond

; CHECK-LABEL: @testqv4floatule
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

define <4 x float> @testqv4floateq(float %c1, float %c2, float %c3, float %c4, <4 x float> %a1, <4 x float> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp eq i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x float> %a1, <4 x float> %a2
  ret <4 x float> %cond

; CHECK-LABEL: @testqv4floateq
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: creqv [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

define <4 x float> @testqv4floatsge(float %c1, float %c2, float %c3, float %c4, <4 x float> %a1, <4 x float> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp sge i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x float> %a1, <4 x float> %a2
  ret <4 x float> %cond

; CHECK-LABEL: @testqv4floatsge
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

define <4 x float> @testqv4floatuge(float %c1, float %c2, float %c3, float %c4, <4 x float> %a1, <4 x float> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp uge i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x float> %a1, <4 x float> %a2
  ret <4 x float> %cond

; CHECK-LABEL: @testqv4floatuge
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

define <4 x float> @testqv4floatsgt(float %c1, float %c2, float %c3, float %c4, <4 x float> %a1, <4 x float> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp sgt i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x float> %a1, <4 x float> %a2
  ret <4 x float> %cond

; CHECK-LABEL: @testqv4floatsgt
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

define <4 x float> @testqv4floatugt(float %c1, float %c2, float %c3, float %c4, <4 x float> %a1, <4 x float> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp ugt i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x float> %a1, <4 x float> %a2
  ret <4 x float> %cond

; CHECK-LABEL: @testqv4floatugt
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

define <4 x float> @testqv4floatne(float %c1, float %c2, float %c3, float %c4, <4 x float> %a1, <4 x float> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp ne i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x float> %a1, <4 x float> %a2
  ret <4 x float> %cond

; CHECK-LABEL: @testqv4floatne
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crxor [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

define <4 x i1> @testqv4i1slt(float %c1, float %c2, float %c3, float %c4, <4 x i1> %a1, <4 x i1> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp slt i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x i1> %a1, <4 x i1> %a2
  ret <4 x i1> %cond

; CHECK-LABEL: @testqv4i1slt
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

define <4 x i1> @testqv4i1ult(float %c1, float %c2, float %c3, float %c4, <4 x i1> %a1, <4 x i1> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp ult i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x i1> %a1, <4 x i1> %a2
  ret <4 x i1> %cond

; CHECK-LABEL: @testqv4i1ult
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

define <4 x i1> @testqv4i1sle(float %c1, float %c2, float %c3, float %c4, <4 x i1> %a1, <4 x i1> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp sle i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x i1> %a1, <4 x i1> %a2
  ret <4 x i1> %cond

; CHECK-LABEL: @testqv4i1sle
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

define <4 x i1> @testqv4i1ule(float %c1, float %c2, float %c3, float %c4, <4 x i1> %a1, <4 x i1> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp ule i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x i1> %a1, <4 x i1> %a2
  ret <4 x i1> %cond

; CHECK-LABEL: @testqv4i1ule
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

define <4 x i1> @testqv4i1eq(float %c1, float %c2, float %c3, float %c4, <4 x i1> %a1, <4 x i1> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp eq i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x i1> %a1, <4 x i1> %a2
  ret <4 x i1> %cond

; CHECK-LABEL: @testqv4i1eq
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: creqv [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

define <4 x i1> @testqv4i1sge(float %c1, float %c2, float %c3, float %c4, <4 x i1> %a1, <4 x i1> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp sge i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x i1> %a1, <4 x i1> %a2
  ret <4 x i1> %cond

; CHECK-LABEL: @testqv4i1sge
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

define <4 x i1> @testqv4i1uge(float %c1, float %c2, float %c3, float %c4, <4 x i1> %a1, <4 x i1> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp uge i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x i1> %a1, <4 x i1> %a2
  ret <4 x i1> %cond

; CHECK-LABEL: @testqv4i1uge
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crorc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

define <4 x i1> @testqv4i1sgt(float %c1, float %c2, float %c3, float %c4, <4 x i1> %a1, <4 x i1> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp sgt i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x i1> %a1, <4 x i1> %a2
  ret <4 x i1> %cond

; CHECK-LABEL: @testqv4i1sgt
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

define <4 x i1> @testqv4i1ugt(float %c1, float %c2, float %c3, float %c4, <4 x i1> %a1, <4 x i1> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp ugt i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x i1> %a1, <4 x i1> %a2
  ret <4 x i1> %cond

; CHECK-LABEL: @testqv4i1ugt
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crandc [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

define <4 x i1> @testqv4i1ne(float %c1, float %c2, float %c3, float %c4, <4 x i1> %a1, <4 x i1> %a2) #1 {
entry:
  %cmp1 = fcmp oeq float %c3, %c4
  %cmp3tmp = fcmp oeq float %c1, %c2
  %cmp3 = icmp ne i1 %cmp3tmp, %cmp1
  %cond = select i1 %cmp3, <4 x i1> %a1, <4 x i1> %a2
  ret <4 x i1> %cond

; CHECK-LABEL: @testqv4i1ne
; CHECK-DAG: fcmpu {{[0-9]+}}, 3, 4
; CHECK-DAG: fcmpu {{[0-9]+}}, 1, 2
; CHECK: crxor [[REG1:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}
; CHECK: bc 12, [[REG1]], .LBB[[BB:[0-9_]+]]
; CHECK: qvfmr 5, 6
; CHECK: .LBB[[BB]]:
; CHECK: qvfmr 1, 5
; CHECK: blr
}

attributes #0 = { nounwind readnone "target-cpu"="pwr7" }
attributes #1 = { nounwind readnone "target-cpu"="a2q" }

