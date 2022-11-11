; RUN: llc -relocation-model=static -verify-machineinstrs -mcpu=pwr7 -O1 -code-model=medium <%s | FileCheck %s
; RUN: llc -relocation-model=static -verify-machineinstrs -mcpu=pwr8 -O1 -code-model=medium <%s | FileCheck %s

; Test peephole optimization for medium code model (32-bit TOC offsets)
; for loading and storing small offsets within aligned values.

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.b4 = type<{ i8, i8, i8, i8 }>
%struct.h2 = type<{ i16, i16 }>

%struct.b8 = type<{ i8, i8, i8, i8, i8, i8, i8, i8 }>
%struct.h4 = type<{ i16, i16, i16, i16 }>
%struct.w2 = type<{ i32, i32 }>

%struct.d2 = type<{ i64, i64 }>
%struct.misalign = type<{ i8, i64 }>

@b4v = dso_local global %struct.b4 <{ i8 1, i8 2, i8 3, i8 4 }>, align 4
@h2v = dso_local global %struct.h2 <{ i16 1, i16 2 }>, align 4

@b8v = dso_local global %struct.b8 <{ i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8 }>, align 8
@h4v = dso_local global %struct.h4 <{ i16 1, i16 2, i16 3, i16 4 }>, align 8
@w2v = dso_local global %struct.w2 <{ i32 1, i32 2 }>, align 8

@d2v = dso_local global %struct.d2 <{ i64 1, i64 2 }>, align 16
@misalign_v = dso_local global %struct.misalign <{ i8 1, i64 2 }>, align 16

; CHECK-LABEL: test_b4:
; CHECK: addis [[REGSTRUCT:[0-9]+]], 2, b4v@toc@ha
; CHECK-DAG: lbz [[REG0_0:[0-9]+]], b4v@toc@l([[REGSTRUCT]])
; CHECK-DAG: lbz [[REG1_0:[0-9]+]], b4v@toc@l+1([[REGSTRUCT]])
; CHECK-DAG: lbz [[REG2_0:[0-9]+]], b4v@toc@l+2([[REGSTRUCT]])
; CHECK-DAG: lbz [[REG3_0:[0-9]+]], b4v@toc@l+3([[REGSTRUCT]])
; CHECK-DAG: addi [[REG0_1:[0-9]+]], [[REG0_0]], 1
; CHECK-DAG: addi [[REG1_1:[0-9]+]], [[REG1_0]], 2
; CHECK-DAG: addi [[REG2_1:[0-9]+]], [[REG2_0]], 3
; CHECK-DAG: addi [[REG3_1:[0-9]+]], [[REG3_0]], 4
; CHECK-DAG: stb [[REG0_1]], b4v@toc@l([[REGSTRUCT]])
; CHECK-DAG: stb [[REG1_1]], b4v@toc@l+1([[REGSTRUCT]])
; CHECK-DAG: stb [[REG2_1]], b4v@toc@l+2([[REGSTRUCT]])
; CHECK-DAG: stb [[REG3_1]], b4v@toc@l+3([[REGSTRUCT]])

define dso_local void @test_b4() nounwind {
entry:
  %0 = load i8, ptr @b4v, align 1
  %inc0 = add nsw i8 %0, 1
  store i8 %inc0, ptr @b4v, align 1
  %1 = load i8, ptr getelementptr inbounds (%struct.b4, ptr @b4v, i32 0, i32 1), align 1
  %inc1 = add nsw i8 %1, 2
  store i8 %inc1, ptr getelementptr inbounds (%struct.b4, ptr @b4v, i32 0, i32 1), align 1
  %2 = load i8, ptr getelementptr inbounds (%struct.b4, ptr @b4v, i32 0, i32 2), align 1
  %inc2 = add nsw i8 %2, 3
  store i8 %inc2, ptr getelementptr inbounds (%struct.b4, ptr @b4v, i32 0, i32 2), align 1
  %3 = load i8, ptr getelementptr inbounds (%struct.b4, ptr @b4v, i32 0, i32 3), align 1
  %inc3 = add nsw i8 %3, 4
  store i8 %inc3, ptr getelementptr inbounds (%struct.b4, ptr @b4v, i32 0, i32 3), align 1
  ret void
}

; CHECK-LABEL: test_h2:
; CHECK: addis [[REGSTRUCT:[0-9]+]], 2, h2v@toc@ha
; CHECK-DAG: lhz [[REG0_0:[0-9]+]], h2v@toc@l([[REGSTRUCT]])
; CHECK-DAG: lhz [[REG1_0:[0-9]+]], h2v@toc@l+2([[REGSTRUCT]])
; CHECK-DAG: addi [[REG0_1:[0-9]+]], [[REG0_0]], 1
; CHECK-DAG: addi [[REG1_1:[0-9]+]], [[REG1_0]], 2
; CHECK-DAG: sth [[REG0_1]], h2v@toc@l([[REGSTRUCT]])
; CHECK-DAG: sth [[REG1_1]], h2v@toc@l+2([[REGSTRUCT]])

define dso_local void @test_h2() nounwind {
entry:
  %0 = load i16, ptr @h2v, align 2
  %inc0 = add nsw i16 %0, 1
  store i16 %inc0, ptr @h2v, align 2
  %1 = load i16, ptr getelementptr inbounds (%struct.h2, ptr @h2v, i32 0, i32 1), align 2
  %inc1 = add nsw i16 %1, 2
  store i16 %inc1, ptr getelementptr inbounds (%struct.h2, ptr @h2v, i32 0, i32 1), align 2
  ret void
}

; CHECK-LABEL: test_h2_optsize:
; CHECK: addis [[REGSTRUCT:[0-9]+]], 2, h2v@toc@ha
; CHECK-DAG: lhz [[REG0_0:[0-9]+]], h2v@toc@l([[REGSTRUCT]])
; CHECK-DAG: lhz [[REG1_0:[0-9]+]], h2v@toc@l+2([[REGSTRUCT]])
; CHECK-DAG: addi [[REG0_1:[0-9]+]], [[REG0_0]], 1
; CHECK-DAG: addi [[REG1_1:[0-9]+]], [[REG1_0]], 2
; CHECK-DAG: sth [[REG0_1]], h2v@toc@l([[REGSTRUCT]])
; CHECK-DAG: sth [[REG1_1]], h2v@toc@l+2([[REGSTRUCT]])
define dso_local void @test_h2_optsize() optsize nounwind {
entry:
  %0 = load i16, ptr @h2v, align 2
  %inc0 = add nsw i16 %0, 1
  store i16 %inc0, ptr @h2v, align 2
  %1 = load i16, ptr getelementptr inbounds (%struct.h2, ptr @h2v, i32 0, i32 1), align 2
  %inc1 = add nsw i16 %1, 2
  store i16 %inc1, ptr getelementptr inbounds (%struct.h2, ptr @h2v, i32 0, i32 1), align 2
  ret void
}

; CHECK-LABEL: test_b8:
; CHECK: addis [[REGSTRUCT:[0-9]+]], 2, b8v@toc@ha
; CHECK-DAG: lbz [[REG0_0:[0-9]+]], b8v@toc@l([[REGSTRUCT]])
; CHECK-DAG: lbz [[REG1_0:[0-9]+]], b8v@toc@l+1([[REGSTRUCT]])
; CHECK-DAG: lbz [[REG2_0:[0-9]+]], b8v@toc@l+2([[REGSTRUCT]])
; CHECK-DAG: lbz [[REG3_0:[0-9]+]], b8v@toc@l+3([[REGSTRUCT]])
; CHECK-DAG: lbz [[REG4_0:[0-9]+]], b8v@toc@l+4([[REGSTRUCT]])
; CHECK-DAG: lbz [[REG5_0:[0-9]+]], b8v@toc@l+5([[REGSTRUCT]])
; CHECK-DAG: lbz [[REG6_0:[0-9]+]], b8v@toc@l+6([[REGSTRUCT]])
; CHECK-DAG: lbz [[REG7_0:[0-9]+]], b8v@toc@l+7([[REGSTRUCT]])
; CHECK-DAG: addi [[REG0_1:[0-9]+]], [[REG0_0]], 1
; CHECK-DAG: addi [[REG1_1:[0-9]+]], [[REG1_0]], 2
; CHECK-DAG: addi [[REG2_1:[0-9]+]], [[REG2_0]], 3
; CHECK-DAG: addi [[REG3_1:[0-9]+]], [[REG3_0]], 4
; CHECK-DAG: addi [[REG4_1:[0-9]+]], [[REG4_0]], 5
; CHECK-DAG: addi [[REG5_1:[0-9]+]], [[REG5_0]], 6
; CHECK-DAG: addi [[REG6_1:[0-9]+]], [[REG6_0]], 7
; CHECK-DAG: addi [[REG7_1:[0-9]+]], [[REG7_0]], 8
; CHECK-DAG: stb [[REG0_1]], b8v@toc@l([[REGSTRUCT]])
; CHECK-DAG: stb [[REG1_1]], b8v@toc@l+1([[REGSTRUCT]])
; CHECK-DAG: stb [[REG2_1]], b8v@toc@l+2([[REGSTRUCT]])
; CHECK-DAG: stb [[REG3_1]], b8v@toc@l+3([[REGSTRUCT]])
; CHECK-DAG: stb [[REG4_1]], b8v@toc@l+4([[REGSTRUCT]])
; CHECK-DAG: stb [[REG5_1]], b8v@toc@l+5([[REGSTRUCT]])
; CHECK-DAG: stb [[REG6_1]], b8v@toc@l+6([[REGSTRUCT]])
; CHECK-DAG: stb [[REG7_1]], b8v@toc@l+7([[REGSTRUCT]])

define dso_local void @test_b8() nounwind {
entry:
  %0 = load i8, ptr @b8v, align 1
  %inc0 = add nsw i8 %0, 1
  store i8 %inc0, ptr @b8v, align 1
  %1 = load i8, ptr getelementptr inbounds (%struct.b8, ptr @b8v, i32 0, i32 1), align 1
  %inc1 = add nsw i8 %1, 2
  store i8 %inc1, ptr getelementptr inbounds (%struct.b8, ptr @b8v, i32 0, i32 1), align 1
  %2 = load i8, ptr getelementptr inbounds (%struct.b8, ptr @b8v, i32 0, i32 2), align 1
  %inc2 = add nsw i8 %2, 3
  store i8 %inc2, ptr getelementptr inbounds (%struct.b8, ptr @b8v, i32 0, i32 2), align 1
  %3 = load i8, ptr getelementptr inbounds (%struct.b8, ptr @b8v, i32 0, i32 3), align 1
  %inc3 = add nsw i8 %3, 4
  store i8 %inc3, ptr getelementptr inbounds (%struct.b8, ptr @b8v, i32 0, i32 3), align 1
  %4 = load i8, ptr getelementptr inbounds (%struct.b8, ptr @b8v, i32 0, i32 4), align 1
  %inc4 = add nsw i8 %4, 5
  store i8 %inc4, ptr getelementptr inbounds (%struct.b8, ptr @b8v, i32 0, i32 4), align 1
  %5 = load i8, ptr getelementptr inbounds (%struct.b8, ptr @b8v, i32 0, i32 5), align 1
  %inc5 = add nsw i8 %5, 6
  store i8 %inc5, ptr getelementptr inbounds (%struct.b8, ptr @b8v, i32 0, i32 5), align 1
  %6 = load i8, ptr getelementptr inbounds (%struct.b8, ptr @b8v, i32 0, i32 6), align 1
  %inc6 = add nsw i8 %6, 7
  store i8 %inc6, ptr getelementptr inbounds (%struct.b8, ptr @b8v, i32 0, i32 6), align 1
  %7 = load i8, ptr getelementptr inbounds (%struct.b8, ptr @b8v, i32 0, i32 7), align 1
  %inc7 = add nsw i8 %7, 8
  store i8 %inc7, ptr getelementptr inbounds (%struct.b8, ptr @b8v, i32 0, i32 7), align 1
  ret void
}

; CHECK-LABEL: test_h4:
; CHECK: addis [[REGSTRUCT:[0-9]+]], 2, h4v@toc@ha
; CHECK-DAG: lhz [[REG0_0:[0-9]+]], h4v@toc@l([[REGSTRUCT]])
; CHECK-DAG: lhz [[REG1_0:[0-9]+]], h4v@toc@l+2([[REGSTRUCT]])
; CHECK-DAG: lhz [[REG2_0:[0-9]+]], h4v@toc@l+4([[REGSTRUCT]])
; CHECK-DAG: lhz [[REG3_0:[0-9]+]], h4v@toc@l+6([[REGSTRUCT]])
; CHECK-DAG: addi [[REG0_1:[0-9]+]], [[REG0_0]], 1
; CHECK-DAG: addi [[REG1_1:[0-9]+]], [[REG1_0]], 2
; CHECK-DAG: addi [[REG2_1:[0-9]+]], [[REG2_0]], 3
; CHECK-DAG: addi [[REG3_1:[0-9]+]], [[REG3_0]], 4
; CHECK-DAG: sth [[REG0_1]], h4v@toc@l([[REGSTRUCT]])
; CHECK-DAG: sth [[REG1_1]], h4v@toc@l+2([[REGSTRUCT]])
; CHECK-DAG: sth [[REG2_1]], h4v@toc@l+4([[REGSTRUCT]])
; CHECK-DAG: sth [[REG3_1]], h4v@toc@l+6([[REGSTRUCT]])

define dso_local void @test_h4() nounwind {
entry:
  %0 = load i16, ptr @h4v, align 2
  %inc0 = add nsw i16 %0, 1
  store i16 %inc0, ptr @h4v, align 2
  %1 = load i16, ptr getelementptr inbounds (%struct.h4, ptr @h4v, i32 0, i32 1), align 2
  %inc1 = add nsw i16 %1, 2
  store i16 %inc1, ptr getelementptr inbounds (%struct.h4, ptr @h4v, i32 0, i32 1), align 2
  %2 = load i16, ptr getelementptr inbounds (%struct.h4, ptr @h4v, i32 0, i32 2), align 2
  %inc2 = add nsw i16 %2, 3
  store i16 %inc2, ptr getelementptr inbounds (%struct.h4, ptr @h4v, i32 0, i32 2), align 2
  %3 = load i16, ptr getelementptr inbounds (%struct.h4, ptr @h4v, i32 0, i32 3), align 2
  %inc3 = add nsw i16 %3, 4
  store i16 %inc3, ptr getelementptr inbounds (%struct.h4, ptr @h4v, i32 0, i32 3), align 2
  ret void
}

; CHECK-LABEL: test_w2:
; CHECK: addis [[REGSTRUCT:[0-9]+]], 2, w2v@toc@ha
; CHECK-DAG: lwz [[REG0_0:[0-9]+]], w2v@toc@l([[REGSTRUCT]])
; CHECK-DAG: lwz [[REG1_0:[0-9]+]], w2v@toc@l+4([[REGSTRUCT]])
; CHECK-DAG: addi [[REG0_1:[0-9]+]], [[REG0_0]], 1
; CHECK-DAG: addi [[REG1_1:[0-9]+]], [[REG1_0]], 2
; CHECK-DAG: stw [[REG0_1]], w2v@toc@l([[REGSTRUCT]])
; CHECK-DAG: stw [[REG1_1]], w2v@toc@l+4([[REGSTRUCT]])

define dso_local void @test_w2() nounwind {
entry:
  %0 = load i32, ptr @w2v, align 4
  %inc0 = add nsw i32 %0, 1
  store i32 %inc0, ptr @w2v, align 4
  %1 = load i32, ptr getelementptr inbounds (%struct.w2, ptr @w2v, i32 0, i32 1), align 4
  %inc1 = add nsw i32 %1, 2
  store i32 %inc1, ptr getelementptr inbounds (%struct.w2, ptr @w2v, i32 0, i32 1), align 4
  ret void
}

; CHECK-LABEL: test_d2:
; CHECK: addis [[REGSTRUCT:[0-9]+]], 2, d2v@toc@ha
; CHECK: addi [[BASEV:[0-9]+]], [[REGSTRUCT]], d2v@toc@l
; CHECK-DAG: ld [[REG0_0:[0-9]+]], d2v@toc@l([[REGSTRUCT]])
; CHECK-DAG: ld [[REG1_0:[0-9]+]], 8([[BASEV]])
; CHECK-DAG: addi [[REG0_1:[0-9]+]], [[REG0_0]], 1
; CHECK-DAG: addi [[REG1_1:[0-9]+]], [[REG1_0]], 2
; CHECK-DAG: std [[REG0_1]], d2v@toc@l([[REGSTRUCT]])
; CHECK-DAG: std [[REG1_1]], 8([[BASEV]])

define dso_local void @test_d2() nounwind {
entry:
  %0 = load i64, ptr @d2v, align 8
  %inc0 = add nsw i64 %0, 1
  store i64 %inc0, ptr @d2v, align 8
  %1 = load i64, ptr getelementptr inbounds (%struct.d2, ptr @d2v, i32 0, i32 1), align 8
  %inc1 = add nsw i64 %1, 2
  store i64 %inc1, ptr getelementptr inbounds (%struct.d2, ptr @d2v, i32 0, i32 1), align 8
  ret void
}

; CHECK-LABEL: test_singleuse:
; CHECK: addis [[REG:[0-9]+]], 2, d2v@toc@ha+8
; CHECK: ld 3, d2v@toc@l+8([[REG]])
define i64 @test_singleuse() nounwind {
entry:
  %0 = load i64, ptr getelementptr inbounds (%struct.d2, ptr @d2v, i32 0, i32 1), align 8
  ret i64 %0
}

; Make sure the optimization fails to fire if the symbol is aligned, but the offset is not.
; CHECK-LABEL: test_misalign
; CHECK: addis [[REGSTRUCT_0:[0-9]+]], 2, misalign_v@toc@ha
; CHECK-DAG: addi [[REGSTRUCT:[0-9]+]], [[REGSTRUCT_0]], misalign_v@toc@l
; CHECK-DAG: li [[OFFSET_REG:[0-9]+]], 1
; CHECK: ldx [[REG0_0:[0-9]+]], [[REGSTRUCT]], [[OFFSET_REG]]
; CHECK: addi [[REG0_1:[0-9]+]], [[REG0_0]], 1
; CHECK: stdx [[REG0_1]], [[REGSTRUCT]], [[OFFSET_REG]]
define dso_local void @test_misalign() nounwind {
entry:
  %0 = load i64, ptr getelementptr inbounds (%struct.misalign, ptr @misalign_v, i32 0, i32 1), align 1
  %inc0 = add nsw i64 %0, 1
  store i64 %inc0, ptr getelementptr inbounds (%struct.misalign, ptr @misalign_v, i32 0, i32 1), align 1
  ret void
}
