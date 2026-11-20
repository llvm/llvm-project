; RUN: opt %loadNPMPolly '-passes=polly-custom<opt-isl;ast>' -plugin-arg=Polly,-polly-print-ast -disable-output < %s | FileCheck %s --check-prefix=CHECK
; RUN: opt %loadNPMPolly '-passes=polly-custom<opt-isl;ast>' -plugin-arg=Polly,-polly-print-ast -disable-output -plugin-arg=Polly,-polly-tile-outer-bands=false < %s | FileCheck %s --check-prefix=OFF
; RUN: opt %loadNPMPolly '-passes=polly-custom<opt-isl;ast>' -plugin-arg=Polly,-polly-print-ast -disable-output -plugin-arg=Polly,-polly-tiling=false -plugin-arg=Polly,-polly-2nd-level-tiling < %s | FileCheck %s --check-prefix=SECOND
; RUN: opt %loadNPMPolly '-passes=polly-custom<opt-isl;ast>' -plugin-arg=Polly,-polly-print-ast -disable-output -plugin-arg=Polly,-polly-2nd-level-tiling < %s | FileCheck %s --check-prefix=TWO
; RUN: opt %loadNPMPolly '-passes=polly-custom<opt-isl;ast>' -plugin-arg=Polly,-polly-print-ast -disable-output -plugin-arg=Polly,-polly-register-tiling < %s | FileCheck %s --check-prefix=OUTER
; RUN: opt %loadNPMPolly '-passes=polly-custom<opt-isl;ast>' -plugin-arg=Polly,-polly-print-ast -disable-output -plugin-arg=Polly,-polly-vectorizer=stripmine < %s | FileCheck %s --check-prefix=OUTER
;
; Tile the outer (i,j) band after the inner (k,l) band has been tiled. The
; bottom-up traversal reaches the outer band with a tiling mark as its child.
; Each original band is tiled once; the generated tile bands are not retiled.
; The one-dimensional inner band in inner1 also leaves the outer band tileable.
;
; inner2: T[i][j][k][l] = T[i-1][j][63-k][63-l] + T[i][j][k-1][l]
;                       + T[i][j][k][l-1];
; inner1: U[i][j][k] = U[i-1][j][63-k] + U[i][j][k-1];
;
; CHECK-LABEL: :: isl ast :: inner2 ::
; CHECK:      // 1st level tiling - Tiles
; CHECK-NEXT: for (int c0 = 0; c0 <= 3; c0 += 1)
; CHECK-NEXT:   for (int c1 = 0; c1 <= 3; c1 += 1) {
; CHECK-NEXT:     // 1st level tiling - Points
; CHECK-NEXT:     for (int c2 = 0; c2 <= min(31, -32 * c0 + 99); c2 += 1)
; CHECK-NEXT:       for (int c3 = 0; c3 <= min(31, -32 * c1 + 98); c3 += 1) {
; CHECK-NEXT:         // 1st level tiling - Tiles
; CHECK-NEXT:         for (int c4 = 0; c4 <= 1; c4 += 1)
; CHECK-NEXT:           for (int c5 = 0; c5 <= 1; c5 += 1) {
; CHECK-NEXT:             // 1st level tiling - Points
; CHECK-NEXT:             for (int c6 = 0; c6 <= min(31, -32 * c4 + 62); c6 += 1)
; CHECK-NEXT:               for (int c7 = 0; c7 <= min(31, -32 * c5 + 62); c7 += 1)
; CHECK-NEXT:                 Stmt_lloop(32 * c1 + c3, 32 * c0 + c2, 32 * c4 + c6, 32 * c5 + c7);
; CHECK-NOT:  // 1st level tiling
; CHECK-LABEL: :: isl ast :: inner1 ::
; CHECK:      // 1st level tiling - Tiles
; CHECK:      // 1st level tiling - Points
; CHECK:      Stmt_kloop(32 * c1 + c3, 32 * c0 + c2, c4);
; CHECK-NOT:  // 1st level tiling
;
; OFF-LABEL: :: isl ast :: inner2 ::
; OFF:      for (int c0 = 0; c0 <= 99; c0 += 1)
; OFF-NEXT:   for (int c1 = 0; c1 <= 98; c1 += 1) {
; OFF-NEXT:     // 1st level tiling - Tiles
; OFF:          Stmt_lloop(c1, c0, 32 * c2 + c4, 32 * c3 + c5);
; OFF-NOT:  // 1st level tiling
; OFF-LABEL: :: isl ast :: inner1 ::
; OFF-NOT:  // 1st level tiling
; OFF:      Stmt_kloop(c1, c0, c2);
; OFF-NOT:  // 1st level tiling
;
; SECOND-LABEL: :: isl ast :: inner2 ::
; SECOND:      // 2nd level tiling - Tiles
; SECOND-NEXT: for (int c0 = 0; c0 <= 6; c0 += 1)
; SECOND-NEXT:   for (int c1 = 0; c1 <= 6; c1 += 1) {
; SECOND:      // 2nd level tiling - Points
; SECOND:      // 2nd level tiling - Tiles
; SECOND:      // 2nd level tiling - Points
; SECOND:      Stmt_lloop(16 * c1 + c3, 16 * c0 + c2, 16 * c4 + c6, 16 * c5 + c7);
; SECOND-NOT:  // 2nd level tiling
; SECOND-LABEL: :: isl ast :: inner1 ::
; SECOND:      // 2nd level tiling - Tiles
; SECOND:      // 2nd level tiling - Points
; SECOND:      Stmt_kloop(16 * c1 + c3, 16 * c0 + c2, c4);
; SECOND-NOT:  // 2nd level tiling
;
; TWO-LABEL: :: isl ast :: inner2 ::
; TWO:      // 1st level tiling - Tiles
; TWO:      // 1st level tiling - Points
; TWO:      // 2nd level tiling - Tiles
; TWO:      // 2nd level tiling - Points
; TWO:      // 1st level tiling - Tiles
; TWO:      // 1st level tiling - Points
; TWO:      // 2nd level tiling - Tiles
; TWO:      // 2nd level tiling - Points
; TWO:      Stmt_lloop(
; TWO-NOT:  // {{1st|2nd}} level tiling
; TWO-LABEL: :: isl ast :: inner1 ::
; TWO:      // 1st level tiling - Tiles
; TWO:      // 1st level tiling - Points
; TWO:      // 2nd level tiling - Tiles
; TWO:      // 2nd level tiling - Points
; TWO:      Stmt_kloop(
; TWO-NOT:  // {{1st|2nd}} level tiling
;
; OUTER-LABEL: :: isl ast :: inner2 ::
; OUTER:      // 1st level tiling - Tiles
; OUTER:      for (int c0 = 0; c0 <= 3; c0 += 1)
; OUTER:        for (int c1 = 0; c1 <= 3; c1 += 1) {
; OUTER:          // 1st level tiling - Points
; OUTER:          for (int c2 = 0; c2 <= min(31, -32 * c0 + 99); c2 += 1)
; OUTER:            for (int c3 = 0; c3 <= min(31, -32 * c1 + 98); c3 += 1) {
; OUTER:              // 1st level tiling - Tiles
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@T = global [100 x [100 x [64 x [64 x double]]]] zeroinitializer
@U = global [100 x [100 x [64 x double]]] zeroinitializer

define void @inner2() {
entry:
  br label %iloop

iloop:
  %i = phi i64 [ 1, %entry ], [ %inext, %iend ]
  %iprev = sub nsw i64 %i, 1
  br label %jloop

jloop:
  %j = phi i64 [ 0, %iloop ], [ %jnext, %jend ]
  br label %kloop

kloop:
  %k = phi i64 [ 1, %jloop ], [ %knext, %kend ]
  %kprev = sub nsw i64 %k, 1
  %kreverse = sub nsw i64 63, %k
  br label %lloop

lloop:
  %l = phi i64 [ 1, %kloop ], [ %lnext, %lloop ]
  %lprev = sub nsw i64 %l, 1
  %lreverse = sub nsw i64 63, %l
  %a = getelementptr inbounds [100 x [100 x [64 x [64 x double]]]], ptr @T, i64 0, i64 %iprev, i64 %j, i64 %kreverse, i64 %lreverse
  %b = getelementptr inbounds [100 x [100 x [64 x [64 x double]]]], ptr @T, i64 0, i64 %i, i64 %j, i64 %kprev, i64 %l
  %c = getelementptr inbounds [100 x [100 x [64 x [64 x double]]]], ptr @T, i64 0, i64 %i, i64 %j, i64 %k, i64 %lprev
  %dst = getelementptr inbounds [100 x [100 x [64 x [64 x double]]]], ptr @T, i64 0, i64 %i, i64 %j, i64 %k, i64 %l
  %av = load double, ptr %a
  %bv = load double, ptr %b
  %cv = load double, ptr %c
  %ab = fadd double %av, %bv
  %sum = fadd double %ab, %cv
  store double %sum, ptr %dst
  %lnext = add nuw nsw i64 %l, 1
  %ldone = icmp eq i64 %lnext, 64
  br i1 %ldone, label %kend, label %lloop

kend:
  %knext = add nuw nsw i64 %k, 1
  %kdone = icmp eq i64 %knext, 64
  br i1 %kdone, label %jend, label %kloop

jend:
  %jnext = add nuw nsw i64 %j, 1
  %jdone = icmp eq i64 %jnext, 100
  br i1 %jdone, label %iend, label %jloop

iend:
  %inext = add nuw nsw i64 %i, 1
  %idone = icmp eq i64 %inext, 100
  br i1 %idone, label %exit, label %iloop

exit:
  ret void
}

define void @inner1() {
entry:
  br label %iloop

iloop:
  %i = phi i64 [ 1, %entry ], [ %inext, %iend ]
  %iprev = sub nsw i64 %i, 1
  br label %jloop

jloop:
  %j = phi i64 [ 0, %iloop ], [ %jnext, %jend ]
  br label %kloop

kloop:
  %k = phi i64 [ 1, %jloop ], [ %knext, %kend ]
  %kprev = sub nsw i64 %k, 1
  %kreverse = sub nsw i64 63, %k
  %a = getelementptr inbounds [100 x [100 x [64 x double]]], ptr @U, i64 0, i64 %iprev, i64 %j, i64 %kreverse
  %b = getelementptr inbounds [100 x [100 x [64 x double]]], ptr @U, i64 0, i64 %i, i64 %j, i64 %kprev
  %dst = getelementptr inbounds [100 x [100 x [64 x double]]], ptr @U, i64 0, i64 %i, i64 %j, i64 %k
  %av = load double, ptr %a
  %bv = load double, ptr %b
  %sum = fadd double %av, %bv
  store double %sum, ptr %dst
  br label %kend

kend:
  %knext = add nuw nsw i64 %k, 1
  %kdone = icmp eq i64 %knext, 64
  br i1 %kdone, label %jend, label %kloop

jend:
  %jnext = add nuw nsw i64 %j, 1
  %jdone = icmp eq i64 %jnext, 100
  br i1 %jdone, label %iend, label %jloop

iend:
  %inext = add nuw nsw i64 %i, 1
  %idone = icmp eq i64 %inext, 100
  br i1 %idone, label %exit, label %iloop

exit:
  ret void
}
