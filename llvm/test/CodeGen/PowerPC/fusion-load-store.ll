; Test if several consecutive loads/stores can be clustered(fused) by scheduler. The
; scheduler will print "Cluster ld/st SU(x) - SU(y)" if SU(x) and SU(y) are fused.

; REQUIRES: asserts
; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr10 \
; RUN:   -mattr=-paired-vector-memops,-pcrelative-memops -verify-misched \
; RUN:   -debug-only=machine-scheduler 2>&1 | FileCheck %s

define i64 @store_i64(ptr nocapture %P, i64 %v) {
entry:
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: store_i64:%bb.0
; CHECK: Cluster ld/st SU([[SU3:[0-9]+]]) - SU([[SU4:[0-9]+]])
; CHECK: Cluster ld/st SU([[SU2:[0-9]+]]) - SU([[SU5:[0-9]+]])
; CHECK: SU([[SU2]]): STD %[[REG:[0-9]+]]:g8rc, 24
; CHECK: SU([[SU3]]): STD %[[REG]]:g8rc, 16
; CHECK: SU([[SU4]]): STD %[[REG]]:g8rc, 8
; CHECK: SU([[SU5]]): STD %[[REG]]:g8rc, 32
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: store_i64:%bb.0
; CHECK: Cluster ld/st SU([[SU0:[0-9]+]]) - SU([[SU1:[0-9]+]])
; CHECK: Cluster ld/st SU([[SU2:[0-9]+]]) - SU([[SU3:[0-9]+]])
; CHECK: SU([[SU0]]): STD renamable $x[[REG:[0-9]+]], 16
; CHECK: SU([[SU1]]): STD renamable $x[[REG]], 8
; CHECK: SU([[SU2]]): STD renamable $x[[REG]], 24
; CHECK: SU([[SU3]]): STD renamable $x[[REG]], 32
  %arrayidx = getelementptr inbounds i64, ptr %P, i64 3
  store i64 %v, ptr %arrayidx
  %arrayidx1 = getelementptr inbounds i64, ptr %P, i64 2
  store i64 %v, ptr %arrayidx1
  %arrayidx2 = getelementptr inbounds i64, ptr %P, i64 1
  store i64 %v, ptr %arrayidx2
  %arrayidx3 = getelementptr inbounds i64, ptr %P, i64 4
  store i64 %v, ptr %arrayidx3
  ret i64 %v
}

define i32 @store_i32(ptr nocapture %P, i32 %v) {
entry:
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: store_i32:%bb.0
; CHECK: Cluster ld/st SU([[SU3:[0-9]+]]) - SU([[SU4:[0-9]+]])
; CHECK: Cluster ld/st SU([[SU2:[0-9]+]]) - SU([[SU5:[0-9]+]])
; CHECK: SU([[SU2]]): STW %[[REG:[0-9]+]].sub_32:g8rc, 52
; CHECK: SU([[SU3]]): STW %[[REG]].sub_32:g8rc, 48
; CHECK: SU([[SU4]]): STW %[[REG]].sub_32:g8rc, 44
; CHECK: SU([[SU5]]): STW %[[REG]].sub_32:g8rc, 56
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: store_i32:%bb.0
; CHECK: Cluster ld/st SU([[SU0:[0-9]+]]) - SU([[SU1:[0-9]+]])
; CHECK: Cluster ld/st SU([[SU2:[0-9]+]]) - SU([[SU3:[0-9]+]])
; CHECK: SU([[SU0]]): STW renamable $r[[REG:[0-9]+]], 48
; CHECK: SU([[SU1]]): STW renamable $r[[REG]], 44
; CHECK: SU([[SU2]]): STW renamable $r[[REG]], 52
; CHECK: SU([[SU3]]): STW renamable $r[[REG]], 56
  %arrayidx = getelementptr inbounds i32, ptr %P, i32 13
  store i32 %v, ptr %arrayidx
  %arrayidx1 = getelementptr inbounds i32, ptr %P, i32 12
  store i32 %v, ptr %arrayidx1
  %arrayidx2 = getelementptr inbounds i32, ptr %P, i32 11
  store i32 %v, ptr %arrayidx2
  %arrayidx3 = getelementptr inbounds i32, ptr %P, i32 14
  store i32 %v, ptr %arrayidx3
  ret i32 %v
}

define void @store_i64_neg(ptr nocapture %P, i64 %v) #0 {
entry:
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: store_i64_neg:%bb.0
; CHECK: Cluster ld/st SU([[SU2:[0-9]+]]) - SU([[SU5:[0-9]+]])
; CHECK: Cluster ld/st SU([[SU3:[0-9]+]]) - SU([[SU4:[0-9]+]])
; CHECK: SU([[SU2]]): STD %[[REG:[0-9]+]]:g8rc, -24
; CHECK: SU([[SU3]]): STD %[[REG]]:g8rc, -8
; CHECK: SU([[SU4]]): STD %[[REG]]:g8rc, -16
; CHECK: SU([[SU5]]): STD %[[REG]]:g8rc, -32
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: store_i64_neg:%bb.0
; CHECK: Cluster ld/st SU([[SU2:[0-9]+]]) - SU([[SU3:[0-9]+]])
; CHECK: Cluster ld/st SU([[SU0:[0-9]+]]) - SU([[SU1:[0-9]+]])
; CHECK: SU([[SU0]]): STD renamable $x[[REG:[0-9]+]], -8
; CHECK: SU([[SU1]]): STD renamable $x[[REG]], -16
; CHECK: SU([[SU2]]): STD renamable $x[[REG]], -24
; CHECK: SU([[SU3]]): STD renamable $x[[REG]], -32
  %arrayidx = getelementptr inbounds i64, ptr %P, i64 -3
  store i64 %v, ptr %arrayidx
  %arrayidx1 = getelementptr inbounds i64, ptr %P, i64 -1
  store i64 %v, ptr %arrayidx1
  %arrayidx2 = getelementptr inbounds i64, ptr %P, i64 -2
  store i64 %v, ptr %arrayidx2
  %arrayidx3 = getelementptr inbounds i64, ptr %P, i64 -4
  store i64 %v, ptr %arrayidx3
  ret void
}

define void @store_i32_neg(ptr nocapture %P, i32 %v) #0 {
entry:
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: store_i32_neg:%bb.0
; CHECK: Cluster ld/st SU([[SU2:[0-9]+]]) - SU([[SU5:[0-9]+]])
; CHECK: Cluster ld/st SU([[SU3:[0-9]+]]) - SU([[SU4:[0-9]+]])
; CHECK: SU([[SU2]]): STW %[[REG:[0-9]+]].sub_32:g8rc, -12
; CHECK: SU([[SU3]]): STW %[[REG]].sub_32:g8rc, -4
; CHECK: SU([[SU4]]): STW %[[REG]].sub_32:g8rc, -8
; CHECK: SU([[SU5]]): STW %[[REG]].sub_32:g8rc, -16
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: store_i32_neg:%bb.0
; CHECK: Cluster ld/st SU([[SU2:[0-9]+]]) - SU([[SU3:[0-9]+]])
; CHECK: Cluster ld/st SU([[SU0:[0-9]+]]) - SU([[SU1:[0-9]+]])
; CHECK:SU([[SU0]]): STW renamable $r[[REG:[0-9]+]], -4
; CHECK:SU([[SU1]]): STW renamable $r[[REG]], -8
; CHECK:SU([[SU2]]): STW renamable $r[[REG]], -12
; CHECK:SU([[SU3]]): STW renamable $r[[REG]], -16
  %arrayidx = getelementptr inbounds i32, ptr %P, i32 -3
  store i32 %v, ptr %arrayidx
  %arrayidx1 = getelementptr inbounds i32, ptr %P, i32 -1
  store i32 %v, ptr %arrayidx1
  %arrayidx2 = getelementptr inbounds i32, ptr %P, i32 -2
  store i32 %v, ptr %arrayidx2
  %arrayidx3 = getelementptr inbounds i32, ptr %P, i32 -4
  store i32 %v, ptr %arrayidx3
  ret void
}

define void @store_double(ptr nocapture %P, double %v)  {
entry:
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: store_double:%bb.0
; CHECK: Cluster ld/st SU([[SU3:[0-9]+]]) - SU([[SU4:[0-9]+]])
; CHECK: Cluster ld/st SU([[SU2:[0-9]+]]) - SU([[SU5:[0-9]+]])
; CHECK: SU([[SU2]]): DFSTOREf64 %[[REG:[0-9]+]]:vsfrc, 24
; CHECK: SU([[SU3]]): DFSTOREf64 %[[REG]]:vsfrc, 8
; CHECK: SU([[SU4]]): DFSTOREf64 %[[REG]]:vsfrc, 16
; CHECK: SU([[SU5]]): DFSTOREf64 %[[REG]]:vsfrc, 32
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: store_double:%bb.0
; CHECK: Cluster ld/st SU([[SU0:[0-9]+]]) - SU([[SU1:[0-9]+]])
; CHECK: Cluster ld/st SU([[SU2:[0-9]+]]) - SU([[SU3:[0-9]+]])
; CHECK: SU([[SU0]]): STFD renamable $f[[REG:[0-9]+]], 8
; CHECK: SU([[SU1]]): STFD renamable $f[[REG]], 16
; CHECK: SU([[SU2]]): STFD renamable $f[[REG]], 24
; CHECK: SU([[SU3]]): STFD renamable $f[[REG]], 32
  %arrayidx = getelementptr inbounds double, ptr %P, i64 3
  store double %v, ptr %arrayidx
  %arrayidx1 = getelementptr inbounds double, ptr %P, i64 1
  store double %v, ptr %arrayidx1
  %arrayidx2 = getelementptr inbounds double, ptr %P, i64 2
  store double %v, ptr %arrayidx2
  %arrayidx3 = getelementptr inbounds double, ptr %P, i64 4
  store double %v, ptr %arrayidx3
  ret void
}

define void @store_float(ptr nocapture %P, float %v)  {
entry:
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: store_float:%bb.0
; CHECK-NOT: Cluster ld/st
; CHECK-NOT: Cluster ld/st
; CHECK: SU([[SU2]]): DFSTOREf32 %[[REG:[0-9]+]]:vssrc, 12
; CHECK: SU([[SU3]]): DFSTOREf32 %[[REG]]:vssrc, 4
; CHECK: SU([[SU4]]): DFSTOREf32 %[[REG]]:vssrc, 8
; CHECK: SU([[SU5]]): DFSTOREf32 %[[REG]]:vssrc, 16
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: store_float:%bb.0
; CHECK-NOT: Cluster ld/st
; CHECK-NOT: Cluster ld/st
; CHECK: SU([[SU0]]): STFS renamable $f[[REG:[0-9]+]], 12
; CHECK: SU([[SU1]]): STFS renamable $f[[REG]], 4
; CHECK: SU([[SU2]]): STFS renamable $f[[REG]], 8
; CHECK: SU([[SU3]]): STFS renamable $f[[REG]], 16
  %arrayidx = getelementptr inbounds float, ptr %P, i64 3
  store float %v, ptr %arrayidx
  %arrayidx1 = getelementptr inbounds float, ptr %P, i64 1
  store float %v, ptr %arrayidx1
  %arrayidx2 = getelementptr inbounds float, ptr %P, i64 2
  store float %v, ptr %arrayidx2
  %arrayidx3 = getelementptr inbounds float, ptr %P, i64 4
  store float %v, ptr %arrayidx3
  ret void
}

; Cannot fuse the store/load if there is volatile in between
define i64 @store_volatile(ptr nocapture %P, i64 %v) {
entry:
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: store_volatile:%bb.0
; CHECK-NOT: Cluster ld/st
; CHECK: SU([[SU2]]): STD %[[REG:[0-9]+]]:g8rc, 24
; CHECK: SU([[SU3]]): STD %[[REG]]:g8rc, 16
; CHECK: SU([[SU4]]): STD %[[REG]]:g8rc, 8
; CHECK: SU([[SU5]]): STD %[[REG]]:g8rc, 32
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: store_volatile:%bb.0
; CHECK-NOT: Cluster ld/st
; CHECK: SU([[SU0]]): STD renamable $x[[REG:[0-9]+]], 24
; CHECK: SU([[SU1]]): STD renamable $x[[REG]], 16
; CHECK: SU([[SU2]]): STD renamable $x[[REG]], 8
; CHECK: SU([[SU3]]): STD renamable $x[[REG]], 32
  %arrayidx = getelementptr inbounds i64, ptr %P, i64 3
  store volatile i64 %v, ptr %arrayidx
  %arrayidx1 = getelementptr inbounds i64, ptr %P, i64 2
  store volatile i64 %v, ptr %arrayidx1
  %arrayidx2 = getelementptr inbounds i64, ptr %P, i64 1
  store volatile i64 %v, ptr %arrayidx2
  %arrayidx3 = getelementptr inbounds i64, ptr %P, i64 4
  store volatile i64 %v, ptr %arrayidx3
  ret i64 %v
}

@p = common local_unnamed_addr global [100 x i32] zeroinitializer, align 4

define void @store_i32_stw_stw8(i32 signext %m, i32 signext %n)  {
entry:
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: store_i32_stw_stw8:%bb.0
; CHECK: Cluster ld/st SU([[SU5:[0-9]+]]) - SU([[SU8:[0-9]+]])
; CHECK: SU([[SU5]]): STW8 %{{[0-9]+}}:g8rc, 24
; CHECK: SU([[SU8]]): STW %{{[0-9]+}}:gprc, 20
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: store_i32_stw_stw8:%bb.0
; CHECK: Cluster ld/st SU([[SU5:[0-9]+]]) - SU([[SU6:[0-9]+]])
; CHECK: SU([[SU5]]): STW8 renamable $x{{[0-9]+}}, 24
; CHECK: SU([[SU6]]): STW renamable $r{{[0-9]+}}, 20
  store i32 9, ptr getelementptr inbounds ([100 x i32], ptr @p, i64 0, i64 6), align 4
  store i32 %n, ptr getelementptr inbounds ([100 x i32], ptr @p, i64 0, i64 7), align 4
  %add = add nsw i32 %n, %m
  store i32 %add, ptr getelementptr inbounds ([100 x i32], ptr @p, i64 0, i64 5), align 4
  ret void
}

define void @store_i32_stw8(i32 signext %m, i32 signext %n)  {
entry:
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: store_i32_stw8:%bb.0
; CHECK: Cluster ld/st SU([[SU4:[0-9]+]]) - SU([[SU5:[0-9]+]])
; CHECK: SU([[SU4]]): STW8 %{{[0-9]+}}:g8rc, 24
; CHECK: SU([[SU5]]): STW8 %{{[0-9]+}}:g8rc, 28
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: store_i32_stw8:%bb.0
; CHECK: Cluster ld/st SU([[SU3:[0-9]+]]) - SU([[SU4:[0-9]+]])
; CHECK: SU([[SU3]]): STW8 renamable $x{{[0-9]+}}, 24
; CHECK: SU([[SU4]]): STW8 renamable $x{{[0-9]+}}, 28
  store i32 9, ptr getelementptr inbounds ([100 x i32], ptr @p, i64 0, i64 6), align 4
  store i32 %n, ptr getelementptr inbounds ([100 x i32], ptr @p, i64 0, i64 7), align 4
  ret void
}

declare void @bar(ptr)

define void @store_frame_index(i32 %a, i32 %b) {
entry:
; CHECK: ********** MI Scheduling **********
; CHECK-LABEL: store_frame_index:%bb.0
; CHECK: Cluster ld/st SU([[SU2:[0-9]+]]) - SU([[SU3:[0-9]+]])
; CHECK: SU([[SU2]]): STD %{{[0-9]+}}:g8rc, 0, %stack.0.buf
; CHECK: SU([[SU3]]): STD %{{[0-9]+}}:g8rc, 8, %stack.0.buf
  %buf = alloca [8 x i64], align 8
  %conv = zext i32 %a to i64
  store i64 %conv, ptr %buf, align 8
  %conv1 = zext i32 %b to i64
  %arrayidx2 = getelementptr inbounds [8 x i64], ptr %buf, i64 0, i64 1
  store i64 %conv1, ptr %arrayidx2, align 8
  call void @bar(ptr nonnull %buf)
  ret void
}
