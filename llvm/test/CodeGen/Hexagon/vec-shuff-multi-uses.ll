; RUN: llc -march=hexagon -mattr=+hvxv68,+hvx-length128b -hexagon-opt-shuffvec=true < %s | FileCheck %s

; This test corresponds to a case where a shufflevector with multiple uses
; was getting incorrectly relocated. The problem was that only one of the uses
; met the safety checks but the pass didn't keep track of it so both
; uses were getting updated at the time of relocation.

; CHECK-NOT:	Relocating after --   {{.*}} = add nuw nsw <128 x i32>

@.str = private unnamed_addr constant [6 x i8] c"vbor \00", align 1

; Function Attrs: nounwind
define dso_local void @vbor(i32 %ntimes, i32 %n, double %ctime, double %dtime, i8* %a, i8* %b, i8* %c, i8* %d, i8* %e, [128 x i8]* %aa, [128 x i8]* %bb, [128 x i8]* %cc) local_unnamed_addr {
entry:
  %s = alloca [128 x i8], align 8
  %0 = getelementptr inbounds [128 x i8], [128 x i8]* %s, i32 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 128, i8* nonnull %0)
  tail call void @init(i32 %n, i8* %a, i8* %b, i8* %c, i8* %d, i8* %e, [128 x i8]* %aa, [128 x i8]* %bb, [128 x i8]* %cc, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str, i32 0, i32 0))
  %call = tail call i32 bitcast (i32 (...)* @second to i32 ()*)()
  %cmp3261 = icmp sgt i32 %n, 0
  %cmp263 = icmp sgt i32 %ntimes, 0
  br i1 %cmp263, label %for.cond2.preheader.preheader, label %for.end141

for.cond2.preheader.preheader:
  %min.iters.check = icmp ult i32 %n, 64
  %min.iters.check272 = icmp ult i32 %n, 128
  %n.vec = and i32 %n, -128
  %cmp.n = icmp eq i32 %n.vec, %n
  %n.vec.remaining = and i32 %n, 64
  %min.epilog.iters.check.not.not = icmp eq i32 %n.vec.remaining, 0
  %n.vec278 = and i32 %n, -64
  %cmp.n281 = icmp eq i32 %n.vec278, %n
  br label %for.cond2.preheader

for.cond2.preheader:                              ; preds = %for.end, %for.cond2.preheader.preheader
  %nl.0264 = phi i32 [ %inc140, %for.end ], [ 0, %for.cond2.preheader.preheader ]
  br i1 %cmp3261, label %iter.check, label %for.end

iter.check:                                       ; preds = %for.cond2.preheader
  br i1 %min.iters.check, label %for.body5.preheader, label %vector.main.loop.iter.check

vector.main.loop.iter.check:                      ; preds = %iter.check
  br i1 %min.iters.check272, label %vec.epilog.ph, label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.main.loop.iter.check
  %index = phi i32 [ %index.next, %vector.body ], [ 0, %vector.main.loop.iter.check ]
  %wide.load = load <128 x i8>, <128 x i8>* poison, align 1
  %wide.load273 = load <128 x i8>, <128 x i8>* poison, align 1
  %wide.load274 = load <128 x i8>, <128 x i8>* poison, align 1
  %wide.load275 = load <128 x i8>, <128 x i8>* poison, align 1
  %wide.load276 = load <128 x i8>, <128 x i8>* poison, align 1
  %wide.load511 = load <128 x i8>, <128 x i8>* poison, align 1
  %1 = zext <128 x i8> %wide.load to <128 x i32>
  %2 = zext <128 x i8> %wide.load273 to <128 x i32>
  %3 = mul nuw nsw <128 x i32> %2, %1
  %4 = zext <128 x i8> %wide.load274 to <128 x i32>
  %5 = zext <128 x i8> %wide.load275 to <128 x i32>
  %6 = zext <128 x i8> %wide.load276 to <128 x i32>
  %7 = zext <128 x i8> %wide.load511 to <128 x i32>
  %8 = add nuw nsw <128 x i32> %6, %5
  %9 = add nuw nsw <128 x i32> %8, %4
  %10 = add nuw nsw <128 x i32> %9, %7
  %11 = mul nuw nsw <128 x i32> %3, %10
  %12 = mul nuw nsw <128 x i32> %4, %1
  %13 = mul nuw nsw <128 x i32> %12, %5
  %14 = mul nuw nsw <128 x i32> %5, %1
  %15 = mul nuw nsw <128 x i32> %6, %1
  %16 = add nuw nsw <128 x i32> %14, %12
  %17 = add nuw nsw <128 x i32> %16, %15
  %18 = mul nuw nsw <128 x i32> %17, %7
  %19 = mul nuw nsw <128 x i32> %16, %6
  %20 = add nuw nsw <128 x i32> %19, %13
  %21 = add nuw nsw <128 x i32> %20, %11
  %22 = add nuw nsw <128 x i32> %21, %18
  %23 = add nuw nsw <128 x i32> %8, %7
  %24 = mul nuw nsw <128 x i32> %23, %4
  %25 = mul nuw nsw <128 x i32> %7, %6
  %26 = add nuw nsw <128 x i32> %24, %25
  %27 = add nuw nsw <128 x i32> %7, %6
  %28 = mul nuw nsw <128 x i32> %27, %5
  %29 = add nuw nsw <128 x i32> %26, %28
  %30 = mul nuw nsw <128 x i32> %29, %2
  %31 = add <128 x i8> %wide.load511, %wide.load276
  %32 = mul <128 x i8> %31, %wide.load275
  %33 = mul <128 x i8> %wide.load511, %wide.load276
  %34 = add <128 x i8> %32, %33
  %35 = shl <128 x i32> %22, <i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24>
  %36 = ashr exact <128 x i32> %35, <i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24>
  %37 = shl <128 x i32> %30, <i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24>
  %38 = ashr exact <128 x i32> %37, <i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24>
  %39 = mul nsw <128 x i32> %36, %38
  %40 = trunc <128 x i32> %39 to <128 x i8>
  %41 = mul <128 x i8> %33, %wide.load274
  %42 = mul <128 x i8> %41, %wide.load275
  %43 = mul <128 x i8> %42, %34
  %44 = mul <128 x i8> %43, %40
  %45 = getelementptr inbounds [128 x i8], [128 x i8]* %s, i32 0, i32 %index
  %46 = bitcast i8* %45 to <128 x i8>*
  store <128 x i8> %44, <128 x i8>* %46, align 8
  %index.next = add nuw i32 %index, 128
  %47 = icmp eq i32 %index.next, %n.vec
  br i1 %47, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body
  br i1 %cmp.n, label %for.end, label %vec.epilog.iter.check

vec.epilog.iter.check:                            ; preds = %middle.block
  br i1 %min.epilog.iters.check.not.not, label %for.body5.preheader, label %vec.epilog.ph

vec.epilog.ph:                                    ; preds = %vec.epilog.iter.check, %vector.main.loop.iter.check
  %vec.epilog.resume.val = phi i32 [ %n.vec, %vec.epilog.iter.check ], [ 0, %vector.main.loop.iter.check ]
  br label %vec.epilog.vector.body

vec.epilog.vector.body:                           ; preds = %vec.epilog.vector.body, %vec.epilog.ph
  %index279 = phi i32 [ %vec.epilog.resume.val, %vec.epilog.ph ], [ %index.next280, %vec.epilog.vector.body ]
  %48 = getelementptr inbounds i8, i8* %a, i32 %index279
  %49 = bitcast i8* %48 to <64 x i8>*
  %wide.load282 = load <64 x i8>, <64 x i8>* %49, align 1
  %50 = getelementptr inbounds i8, i8* %b, i32 %index279
  %51 = bitcast i8* %50 to <64 x i8>*
  %wide.load283 = load <64 x i8>, <64 x i8>* %51, align 1
  %52 = getelementptr inbounds i8, i8* %c, i32 %index279
  %53 = bitcast i8* %52 to <64 x i8>*
  %wide.load284 = load <64 x i8>, <64 x i8>* %53, align 1
  %54 = getelementptr inbounds i8, i8* %d, i32 %index279
  %55 = bitcast i8* %54 to <64 x i8>*
  %wide.load285 = load <64 x i8>, <64 x i8>* %55, align 1
  %56 = getelementptr inbounds i8, i8* %e, i32 %index279
  %57 = bitcast i8* %56 to <64 x i8>*
  %wide.load286 = load <64 x i8>, <64 x i8>* %57, align 1
  %wide.load312 = load <64 x i8>, <64 x i8>* poison, align 1
  %58 = zext <64 x i8> %wide.load282 to <64 x i32>
  %59 = zext <64 x i8> %wide.load283 to <64 x i32>
  %60 = mul nuw nsw <64 x i32> %59, %58
  %61 = zext <64 x i8> %wide.load284 to <64 x i32>
  %62 = zext <64 x i8> %wide.load285 to <64 x i32>
  %63 = zext <64 x i8> %wide.load286 to <64 x i32>
  %64 = zext <64 x i8> %wide.load312 to <64 x i32>
  %65 = add nuw nsw <64 x i32> %63, %62
  %66 = add nuw nsw <64 x i32> %65, %61
  %67 = add nuw nsw <64 x i32> %66, %64
  %68 = mul nuw nsw <64 x i32> %60, %67
  %69 = mul nuw nsw <64 x i32> %61, %58
  %70 = mul nuw nsw <64 x i32> %69, %62
  %71 = mul nuw nsw <64 x i32> %62, %58
  %72 = mul nuw nsw <64 x i32> %63, %58
  %73 = add nuw nsw <64 x i32> %71, %69
  %74 = add nuw nsw <64 x i32> %73, %72
  %75 = mul nuw nsw <64 x i32> %74, %64
  %76 = mul nuw nsw <64 x i32> %73, %63
  %77 = add nuw nsw <64 x i32> %76, %70
  %78 = add nuw nsw <64 x i32> %77, %68
  %79 = add nuw nsw <64 x i32> %78, %75
  %80 = add nuw nsw <64 x i32> %65, %64
  %81 = mul nuw nsw <64 x i32> %80, %61
  %82 = mul nuw nsw <64 x i32> %64, %63
  %83 = add nuw nsw <64 x i32> %81, %82
  %84 = add nuw nsw <64 x i32> %64, %63
  %85 = mul nuw nsw <64 x i32> %84, %62
  %86 = add nuw nsw <64 x i32> %83, %85
  %87 = mul nuw nsw <64 x i32> %86, %59
  %88 = add <64 x i8> %wide.load312, %wide.load286
  %89 = mul <64 x i8> %88, %wide.load285
  %90 = mul <64 x i8> %wide.load312, %wide.load286
  %91 = add <64 x i8> %89, %90
  %92 = shl <64 x i32> %79, <i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24>
  %93 = ashr exact <64 x i32> %92, <i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24>
  %94 = shl <64 x i32> %87, <i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24>
  %95 = ashr exact <64 x i32> %94, <i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24, i32 24>
  %96 = mul nsw <64 x i32> %93, %95
  %97 = trunc <64 x i32> %96 to <64 x i8>
  %98 = mul <64 x i8> %90, %wide.load284
  %99 = mul <64 x i8> %98, %wide.load285
  %100 = mul <64 x i8> %99, %91
  %101 = mul <64 x i8> %100, %97
  %102 = getelementptr inbounds [128 x i8], [128 x i8]* %s, i32 0, i32 %index279
  %103 = bitcast i8* %102 to <64 x i8>*
  store <64 x i8> %101, <64 x i8>* %103, align 8
  %index.next280 = add nuw i32 %index279, 64
  %104 = icmp eq i32 %index.next280, %n.vec278
  br i1 %104, label %vec.epilog.middle.block, label %vec.epilog.vector.body

vec.epilog.middle.block:                          ; preds = %vec.epilog.vector.body
  br i1 %cmp.n281, label %for.end, label %for.body5.preheader

for.body5.preheader:                              ; preds = %vec.epilog.middle.block, %vec.epilog.iter.check, %iter.check
  %i.0262.ph = phi i32 [ 0, %iter.check ], [ %n.vec, %vec.epilog.iter.check ], [ %n.vec278, %vec.epilog.middle.block ]
  br label %for.body5

for.body5:                                        ; preds = %for.body5, %for.body5.preheader
  %i.0262 = phi i32 [ %inc, %for.body5 ], [ %i.0262.ph, %for.body5.preheader ]
  %arrayidx = getelementptr inbounds i8, i8* %a, i32 %i.0262
  %105 = load i8, i8* %arrayidx, align 1
  %arrayidx6 = getelementptr inbounds i8, i8* %b, i32 %i.0262
  %106 = load i8, i8* %arrayidx6, align 1
  %arrayidx7 = getelementptr inbounds i8, i8* %c, i32 %i.0262
  %107 = load i8, i8* %arrayidx7, align 1
  %arrayidx8 = getelementptr inbounds i8, i8* %d, i32 %i.0262
  %108 = load i8, i8* %arrayidx8, align 1
  %arrayidx9 = getelementptr inbounds i8, i8* %e, i32 %i.0262
  %109 = load i8, i8* %arrayidx9, align 1
  %arrayidx11 = getelementptr inbounds [128 x i8], [128 x i8]* %aa, i32 %i.0262, i32 0
  %110 = load i8, i8* %arrayidx11, align 1
  %conv12266 = zext i8 %105 to i32
  %conv13267 = zext i8 %106 to i32
  %mul = mul nuw nsw i32 %conv13267, %conv12266
  %conv14268 = zext i8 %107 to i32
  %conv19269 = zext i8 %108 to i32
  %conv24270 = zext i8 %109 to i32
  %conv30271 = zext i8 %110 to i32
  %mul20243 = add nuw nsw i32 %conv24270, %conv19269
  %mul25244 = add nuw nsw i32 %mul20243, %conv14268
  %mul31245 = add nuw nsw i32 %mul25244, %conv30271
  %add32 = mul nuw nsw i32 %mul, %mul31245
  %mul35 = mul nuw nsw i32 %conv14268, %conv12266
  %mul37 = mul nuw nsw i32 %mul35, %conv19269
  %mul53 = mul nuw nsw i32 %conv19269, %conv12266
  %mul67 = mul nuw nsw i32 %conv24270, %conv12266
  %reass.add = add nuw nsw i32 %mul53, %mul35
  %reass.add250 = add nuw nsw i32 %reass.add, %mul67
  %reass.mul = mul nuw nsw i32 %reass.add250, %conv30271
  %reass.mul252 = mul nuw nsw i32 %reass.add, %conv24270
  %add56 = add nuw nsw i32 %reass.mul252, %mul37
  %add62 = add nuw nsw i32 %add56, %add32
  %add68 = add nuw nsw i32 %add62, %reass.mul
  %mul85247 = add nuw nsw i32 %mul20243, %conv30271
  %add86 = mul nuw nsw i32 %mul85247, %conv14268
  %mul103 = mul nuw nsw i32 %conv30271, %conv24270
  %reass.add253 = add nuw nsw i32 %add86, %mul103
  %reass.add255 = add nuw nsw i32 %conv30271, %conv24270
  %reass.mul256 = mul nuw nsw i32 %reass.add255, %conv19269
  %reass.add259 = add nuw nsw i32 %reass.add253, %reass.mul256
  %reass.mul260 = mul nuw nsw i32 %reass.add259, %conv13267
  %mul115248 = add i8 %110, %109
  %add116 = mul i8 %mul115248, %108
  %mul121 = mul i8 %110, %109
  %reass.add257 = add i8 %add116, %mul121
  %sext = shl i32 %add68, 24
  %conv130 = ashr exact i32 %sext, 24
  %sext249 = shl i32 %reass.mul260, 24
  %conv131 = ashr exact i32 %sext249, 24
  %mul132 = mul nsw i32 %conv130, %conv131
  %111 = trunc i32 %mul132 to i8
  %112 = mul i8 %mul121, %107
  %mul126 = mul i8 %112, %108
  %mul128 = mul i8 %mul126, %reass.add257
  %conv137 = mul i8 %mul128, %111
  %arrayidx138 = getelementptr inbounds [128 x i8], [128 x i8]* %s, i32 0, i32 %i.0262
  store i8 %conv137, i8* %arrayidx138, align 1
  %inc = add nuw nsw i32 %i.0262, 1
  %exitcond.not = icmp eq i32 %inc, %n
  br i1 %exitcond.not, label %for.end, label %for.body5

for.end:                                          ; preds = %for.body5, %vec.epilog.middle.block, %middle.block, %for.cond2.preheader
  tail call void @dummy(i32 %n, i8* %a, i8* %b, i8* %c, i8* %d, i8* %e, [128 x i8]* %aa, [128 x i8]* %bb, [128 x i8]* %cc, i8 signext 1)
  %inc140 = add nuw nsw i32 %nl.0264, 1
  %exitcond265.not = icmp eq i32 %inc140, %ntimes
  br i1 %exitcond265.not, label %for.end141, label %for.cond2.preheader

for.end141:                                       ; preds = %for.end, %entry
  %conv = sitofp i32 %call to double
  %call142 = tail call i32 bitcast (i32 (...)* @second to i32 ()*)()
  %conv143 = sitofp i32 %call142 to double
  %sub = fsub double %conv143, %conv
  %sub144 = fsub double %sub, %ctime
  %conv145 = sitofp i32 %ntimes to double
  %mul146 = fmul double %conv145, %dtime
  %sub147 = fsub double %sub144, %mul146
  %call148 = call i64 @cs1d(i32 %n, i8* nonnull %0)
  %mul149 = mul nsw i32 %n, %ntimes
  call void @check(i64 %call148, i32 %mul149, i32 %n, double %sub147, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str, i32 0, i32 0))
  call void @llvm.lifetime.end.p0i8(i64 128, i8* nonnull %0)
  ret void
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)

declare dso_local void @init(i32, i8*, i8*, i8*, i8*, i8*, [128 x i8]*, [128 x i8]*, [128 x i8]*, i8*) local_unnamed_addr

declare dso_local i32 @second(...) local_unnamed_addr

declare dso_local void @dummy(i32, i8*, i8*, i8*, i8*, i8*, [128 x i8]*, [128 x i8]*, [128 x i8]*, i8 signext) local_unnamed_addr

declare dso_local i64 @cs1d(i32, i8*) local_unnamed_addr

declare dso_local void @check(i64, i32, i32, double, i8*) local_unnamed_addr

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)
