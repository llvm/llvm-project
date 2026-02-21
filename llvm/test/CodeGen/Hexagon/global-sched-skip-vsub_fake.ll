; RUN: llc -march=hexagon < %s
; REQUIRES: asserts
;
; Check that the compiler does not crash. It was asserting because function
; getMinimalPhysRegClass() was called with vsub_fake and vsub_fake does not
; belong to any target register class.

target triple = "hexagon"

@b = common dso_local local_unnamed_addr global i32 0, align 4
@a = common dso_local local_unnamed_addr global i16 0, align 2

define dso_local void @d(i32 noundef %e, ptr nocapture noundef writeonly %g, i16 noundef signext %0, i32 noundef %1) local_unnamed_addr {
entry:
  %cmp.not5 = icmp slt i32 %e, 1
  br i1 %cmp.not5, label %for.end, label %vector.body

vector.body:                                      ; preds = %entry
  %n.rnd.up = add nuw i32 %e, 63
  %n.vec = and i32 %n.rnd.up, -64
  %trip.count.minus.1 = add nsw i32 %e, -1
  %broadcast.splatinsert8 = insertelement <64 x i32> poison, i32 %trip.count.minus.1, i64 0
  %broadcast.splat9 = shufflevector <64 x i32> %broadcast.splatinsert8, <64 x i32> poison, <64 x i32> zeroinitializer
  %invariant.gep = getelementptr i8, ptr %g, i32 128
  call void @d.llvmint.extracted_region.llvmint.1.0_i32_0(<64 x i32> %broadcast.splat9, ptr %g, ptr %invariant.gep, i32 %n.vec)
  br label %for.end

for.end:                                          ; preds = %vector.body, %entry
  ret void
}

define dso_local void @h(ptr nocapture noundef writeonly %e, ptr nocapture noundef writeonly %g) local_unnamed_addr {
entry:
  %call = tail call i32 @c()
  %tobool.not = icmp eq i32 %call, 0
  %0 = load i32, ptr @b, align 4
  %cmp.not5.i20 = icmp slt i32 %0, 1
  br i1 %tobool.not, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  br i1 %cmp.not5.i20, label %if.end, label %vector.body347

vector.body347:                                   ; preds = %if.then
  %n.rnd.up = add nuw i32 %0, 63
  %n.vec = and i32 %n.rnd.up, -64
  %trip.count.minus.1 = add nsw i32 %0, -1
  %broadcast.splatinsert63 = insertelement <64 x i32> poison, i32 %trip.count.minus.1, i64 0
  %broadcast.splat64 = shufflevector <64 x i32> %broadcast.splatinsert63, <64 x i32> poison, <64 x i32> zeroinitializer
  call void @d.llvmint.extracted_region(i32 0, <64 x i32> %broadcast.splat64, ptr nonnull @a, ptr getelementptr (i8, ptr @a, i32 128), i32 %n.vec)
  %invariant.gep = getelementptr i8, ptr %e, i32 128
  call void @d.llvmint.extracted_region.llvmint.1.0_i32_0(<64 x i32> %broadcast.splat64, ptr %e, ptr %invariant.gep, i32 %n.vec)
  %invariant.gep928 = getelementptr i8, ptr %g, i32 128
  call void @d.llvmint.extracted_region.llvmint.1.0_i32_0(<64 x i32> %broadcast.splat64, ptr %g, ptr %invariant.gep928, i32 %n.vec)
  br label %if.end

if.else:                                          ; preds = %entry
  br i1 %cmp.not5.i20, label %d.exit42.thread, label %for.body.i28

d.exit42.thread:                                  ; preds = %if.else
  %arrayidx.i3154 = getelementptr inbounds i8, ptr %e, i32 2
  store i16 1, ptr %arrayidx.i3154, align 2
  %arrayidx.i31.155 = getelementptr inbounds i8, ptr %e, i32 4
  store i16 2, ptr %arrayidx.i31.155, align 2
  br label %if.end

for.body.i28:                                     ; preds = %if.else
  %n.rnd.up488 = add nuw i32 %0, 63
  %n.vec490 = and i32 %n.rnd.up488, -64
  %trip.count.minus.1493 = add nsw i32 %0, -1
  %broadcast.splatinsert501 = insertelement <64 x i32> poison, i32 %trip.count.minus.1493, i64 0
  %broadcast.splat502 = shufflevector <64 x i32> %broadcast.splatinsert501, <64 x i32> poison, <64 x i32> zeroinitializer
  call void @d.llvmint.extracted_region(i32 0, <64 x i32> %broadcast.splat502, ptr nonnull @a, ptr getelementptr (i8, ptr @a, i32 128), i32 %n.vec490)
  %arrayidx.i31 = getelementptr inbounds i8, ptr %e, i32 2
  store i16 1, ptr %arrayidx.i31, align 2
  %arrayidx.i31.1 = getelementptr inbounds i8, ptr %e, i32 4
  store i16 2, ptr %arrayidx.i31.1, align 2
  %cmp.not5.i35 = icmp eq i32 %0, 1
  br i1 %cmp.not5.i35, label %vector.body788, label %d.exit42

d.exit42:                                         ; preds = %for.body.i28
  %arrayidx = getelementptr inbounds i16, ptr %e, i32 %0
  %div60 = lshr i32 %0, 1
  %n.rnd.up635 = add nuw nsw i32 %div60, 63
  %n.vec637 = and i32 %n.rnd.up635, 2147483584
  %trip.count.minus.1640 = add nsw i32 %div60, -1
  %broadcast.splatinsert648 = insertelement <64 x i32> poison, i32 %trip.count.minus.1640, i64 0
  %broadcast.splat649 = shufflevector <64 x i32> %broadcast.splatinsert648, <64 x i32> poison, <64 x i32> zeroinitializer
  %invariant.gep931 = getelementptr i8, ptr %arrayidx, i32 128
  call void @d.llvmint.extracted_region.llvmint.1.0_i32_0(<64 x i32> %broadcast.splat649, ptr nonnull %arrayidx, ptr %invariant.gep931, i32 %n.vec637)
  br label %vector.body788

vector.body788:                                   ; preds = %for.body.i28, %d.exit42
  %invariant.gep933 = getelementptr i8, ptr %g, i32 128
  call void @d.llvmint.extracted_region.llvmint.1.0_i32_0(<64 x i32> %broadcast.splat502, ptr %g, ptr %invariant.gep933, i32 %n.vec490)
  br label %if.end

if.end:                                           ; preds = %vector.body788, %vector.body347, %d.exit42.thread, %if.then
  ret void
}

declare dso_local i32 @c(...) local_unnamed_addr

define internal void @d.llvmint.extracted_region.llvmint.1.0_i32_0(<64 x i32> %0, ptr %1, ptr %2, i32 %3)  {
  tail call void @d.llvmint.extracted_region(i32 0, <64 x i32> %0, ptr %1, ptr %2, i32 %3)
  ret void
}

define internal void @d.llvmint.extracted_region(i32 %0, <64 x i32> %1, ptr %2, ptr %3, i32 %4)  {
entry:
  br label %vector.body.extracted_entry

vector.body.extracted_entry:                      ; preds = %entry, %pred.store.continue135
  %index.extracted = phi i32 [ %index.next, %pred.store.continue135 ], [ %0, %entry ]
  %5 = trunc i32 %index.extracted to i16
  %broadcast.splatinsert = insertelement <64 x i32> poison, i32 %index.extracted, i64 0
  %broadcast.splat = shufflevector <64 x i32> %broadcast.splatinsert, <64 x i32> poison, <64 x i32> zeroinitializer
  %vec.iv = or disjoint <64 x i32> %broadcast.splat, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  %6 = icmp ule <64 x i32> %vec.iv, %1
  %7 = extractelement <64 x i1> %6, i64 0
  br i1 %7, label %pred.store.if, label %pred.store.continue

pred.store.if:                                    ; preds = %vector.body.extracted_entry
  %8 = or disjoint i16 %5, 1
  %offset.idx = or disjoint i32 %index.extracted, 1
  %9 = getelementptr inbounds i16, ptr %2, i32 %offset.idx
  store i16 %8, ptr %9, align 2
  br label %pred.store.continue

pred.store.continue:                              ; preds = %pred.store.if, %vector.body.extracted_entry
  %10 = extractelement <64 x i1> %6, i64 1
  br i1 %10, label %pred.store.if10, label %pred.store.continue11

pred.store.if10:                                  ; preds = %pred.store.continue
  %11 = or disjoint i32 %index.extracted, 2
  %12 = getelementptr inbounds i16, ptr %2, i32 %11
  %13 = or disjoint i16 %5, 2
  store i16 %13, ptr %12, align 2
  br label %pred.store.continue11

pred.store.continue11:                            ; preds = %pred.store.if10, %pred.store.continue
  %14 = extractelement <64 x i1> %6, i64 2
  br i1 %14, label %pred.store.if12, label %pred.store.continue13

pred.store.if12:                                  ; preds = %pred.store.continue11
  %15 = or disjoint i32 %index.extracted, 3
  %16 = getelementptr inbounds i16, ptr %2, i32 %15
  %17 = or disjoint i16 %5, 3
  store i16 %17, ptr %16, align 2
  br label %pred.store.continue13

pred.store.continue13:                            ; preds = %pred.store.if12, %pred.store.continue11
  %18 = extractelement <64 x i1> %6, i64 3
  br i1 %18, label %pred.store.if14, label %pred.store.continue15

pred.store.if14:                                  ; preds = %pred.store.continue13
  %19 = or disjoint i32 %index.extracted, 4
  %20 = getelementptr inbounds i16, ptr %2, i32 %19
  %21 = or disjoint i16 %5, 4
  store i16 %21, ptr %20, align 2
  br label %pred.store.continue15

pred.store.continue15:                            ; preds = %pred.store.if14, %pred.store.continue13
  %22 = extractelement <64 x i1> %6, i64 4
  br i1 %22, label %pred.store.if16, label %pred.store.continue17

pred.store.if16:                                  ; preds = %pred.store.continue15
  %23 = or disjoint i32 %index.extracted, 5
  %24 = getelementptr inbounds i16, ptr %2, i32 %23
  %25 = or disjoint i16 %5, 5
  store i16 %25, ptr %24, align 2
  br label %pred.store.continue17

pred.store.continue17:                            ; preds = %pred.store.if16, %pred.store.continue15
  %26 = extractelement <64 x i1> %6, i64 5
  br i1 %26, label %pred.store.if18, label %pred.store.continue19

pred.store.if18:                                  ; preds = %pred.store.continue17
  %27 = or disjoint i32 %index.extracted, 6
  %28 = getelementptr inbounds i16, ptr %2, i32 %27
  %29 = or disjoint i16 %5, 6
  store i16 %29, ptr %28, align 2
  br label %pred.store.continue19

pred.store.continue19:                            ; preds = %pred.store.if18, %pred.store.continue17
  %30 = extractelement <64 x i1> %6, i64 6
  br i1 %30, label %pred.store.if20, label %pred.store.continue21

pred.store.if20:                                  ; preds = %pred.store.continue19
  %31 = or disjoint i32 %index.extracted, 7
  %32 = getelementptr inbounds i16, ptr %2, i32 %31
  %33 = or disjoint i16 %5, 7
  store i16 %33, ptr %32, align 2
  br label %pred.store.continue21

pred.store.continue21:                            ; preds = %pred.store.if20, %pred.store.continue19
  %34 = extractelement <64 x i1> %6, i64 7
  br i1 %34, label %pred.store.if22, label %pred.store.continue23

pred.store.if22:                                  ; preds = %pred.store.continue21
  %35 = or disjoint i32 %index.extracted, 8
  %36 = getelementptr inbounds i16, ptr %2, i32 %35
  %37 = or disjoint i16 %5, 8
  store i16 %37, ptr %36, align 2
  br label %pred.store.continue23

pred.store.continue23:                            ; preds = %pred.store.if22, %pred.store.continue21
  %38 = extractelement <64 x i1> %6, i64 8
  br i1 %38, label %pred.store.if24, label %pred.store.continue25

pred.store.if24:                                  ; preds = %pred.store.continue23
  %39 = or disjoint i32 %index.extracted, 9
  %40 = getelementptr inbounds i16, ptr %2, i32 %39
  %41 = or disjoint i16 %5, 9
  store i16 %41, ptr %40, align 2
  br label %pred.store.continue25

pred.store.continue25:                            ; preds = %pred.store.if24, %pred.store.continue23
  %42 = extractelement <64 x i1> %6, i64 9
  br i1 %42, label %pred.store.if26, label %pred.store.continue27

pred.store.if26:                                  ; preds = %pred.store.continue25
  %43 = or disjoint i32 %index.extracted, 10
  %44 = getelementptr inbounds i16, ptr %2, i32 %43
  %45 = or disjoint i16 %5, 10
  store i16 %45, ptr %44, align 2
  br label %pred.store.continue27

pred.store.continue27:                            ; preds = %pred.store.if26, %pred.store.continue25
  %46 = extractelement <64 x i1> %6, i64 10
  br i1 %46, label %pred.store.if28, label %pred.store.continue29

pred.store.if28:                                  ; preds = %pred.store.continue27
  %47 = or disjoint i32 %index.extracted, 11
  %48 = getelementptr inbounds i16, ptr %2, i32 %47
  %49 = or disjoint i16 %5, 11
  store i16 %49, ptr %48, align 2
  br label %pred.store.continue29

pred.store.continue29:                            ; preds = %pred.store.if28, %pred.store.continue27
  %50 = extractelement <64 x i1> %6, i64 11
  br i1 %50, label %pred.store.if30, label %pred.store.continue31

pred.store.if30:                                  ; preds = %pred.store.continue29
  %51 = or disjoint i32 %index.extracted, 12
  %52 = getelementptr inbounds i16, ptr %2, i32 %51
  %53 = or disjoint i16 %5, 12
  store i16 %53, ptr %52, align 2
  br label %pred.store.continue31

pred.store.continue31:                            ; preds = %pred.store.if30, %pred.store.continue29
  %54 = extractelement <64 x i1> %6, i64 12
  br i1 %54, label %pred.store.if32, label %pred.store.continue33

pred.store.if32:                                  ; preds = %pred.store.continue31
  %55 = or disjoint i32 %index.extracted, 13
  %56 = getelementptr inbounds i16, ptr %2, i32 %55
  %57 = or disjoint i16 %5, 13
  store i16 %57, ptr %56, align 2
  br label %pred.store.continue33

pred.store.continue33:                            ; preds = %pred.store.if32, %pred.store.continue31
  %58 = extractelement <64 x i1> %6, i64 13
  br i1 %58, label %pred.store.if34, label %pred.store.continue35

pred.store.if34:                                  ; preds = %pred.store.continue33
  %59 = or disjoint i32 %index.extracted, 14
  %60 = getelementptr inbounds i16, ptr %2, i32 %59
  %61 = or disjoint i16 %5, 14
  store i16 %61, ptr %60, align 2
  br label %pred.store.continue35

pred.store.continue35:                            ; preds = %pred.store.if34, %pred.store.continue33
  %62 = extractelement <64 x i1> %6, i64 14
  br i1 %62, label %pred.store.if36, label %pred.store.continue37

pred.store.if36:                                  ; preds = %pred.store.continue35
  %63 = or disjoint i32 %index.extracted, 15
  %64 = getelementptr inbounds i16, ptr %2, i32 %63
  %65 = or disjoint i16 %5, 15
  store i16 %65, ptr %64, align 2
  br label %pred.store.continue37

pred.store.continue37:                            ; preds = %pred.store.if36, %pred.store.continue35
  %66 = extractelement <64 x i1> %6, i64 15
  br i1 %66, label %pred.store.if38, label %pred.store.continue39

pred.store.if38:                                  ; preds = %pred.store.continue37
  %67 = or disjoint i32 %index.extracted, 16
  %68 = getelementptr inbounds i16, ptr %2, i32 %67
  %69 = or disjoint i16 %5, 16
  store i16 %69, ptr %68, align 2
  br label %pred.store.continue39

pred.store.continue39:                            ; preds = %pred.store.if38, %pred.store.continue37
  %70 = extractelement <64 x i1> %6, i64 16
  br i1 %70, label %pred.store.if40, label %pred.store.continue41

pred.store.if40:                                  ; preds = %pred.store.continue39
  %71 = or disjoint i32 %index.extracted, 17
  %72 = getelementptr inbounds i16, ptr %2, i32 %71
  %73 = or disjoint i16 %5, 17
  store i16 %73, ptr %72, align 2
  br label %pred.store.continue41

pred.store.continue41:                            ; preds = %pred.store.if40, %pred.store.continue39
  %74 = extractelement <64 x i1> %6, i64 17
  br i1 %74, label %pred.store.if42, label %pred.store.continue43

pred.store.if42:                                  ; preds = %pred.store.continue41
  %75 = or disjoint i32 %index.extracted, 18
  %76 = getelementptr inbounds i16, ptr %2, i32 %75
  %77 = or disjoint i16 %5, 18
  store i16 %77, ptr %76, align 2
  br label %pred.store.continue43

pred.store.continue43:                            ; preds = %pred.store.if42, %pred.store.continue41
  %78 = extractelement <64 x i1> %6, i64 18
  br i1 %78, label %pred.store.if44, label %pred.store.continue45

pred.store.if44:                                  ; preds = %pred.store.continue43
  %79 = or disjoint i32 %index.extracted, 19
  %80 = getelementptr inbounds i16, ptr %2, i32 %79
  %81 = or disjoint i16 %5, 19
  store i16 %81, ptr %80, align 2
  br label %pred.store.continue45

pred.store.continue45:                            ; preds = %pred.store.if44, %pred.store.continue43
  %82 = extractelement <64 x i1> %6, i64 19
  br i1 %82, label %pred.store.if46, label %pred.store.continue47

pred.store.if46:                                  ; preds = %pred.store.continue45
  %83 = or disjoint i32 %index.extracted, 20
  %84 = getelementptr inbounds i16, ptr %2, i32 %83
  %85 = or disjoint i16 %5, 20
  store i16 %85, ptr %84, align 2
  br label %pred.store.continue47

pred.store.continue47:                            ; preds = %pred.store.if46, %pred.store.continue45
  %86 = extractelement <64 x i1> %6, i64 20
  br i1 %86, label %pred.store.if48, label %pred.store.continue49

pred.store.if48:                                  ; preds = %pred.store.continue47
  %87 = or disjoint i32 %index.extracted, 21
  %88 = getelementptr inbounds i16, ptr %2, i32 %87
  %89 = or disjoint i16 %5, 21
  store i16 %89, ptr %88, align 2
  br label %pred.store.continue49

pred.store.continue49:                            ; preds = %pred.store.if48, %pred.store.continue47
  %90 = extractelement <64 x i1> %6, i64 21
  br i1 %90, label %pred.store.if50, label %pred.store.continue51

pred.store.if50:                                  ; preds = %pred.store.continue49
  %91 = or disjoint i32 %index.extracted, 22
  %92 = getelementptr inbounds i16, ptr %2, i32 %91
  %93 = or disjoint i16 %5, 22
  store i16 %93, ptr %92, align 2
  br label %pred.store.continue51

pred.store.continue51:                            ; preds = %pred.store.if50, %pred.store.continue49
  %94 = extractelement <64 x i1> %6, i64 22
  br i1 %94, label %pred.store.if52, label %pred.store.continue53

pred.store.if52:                                  ; preds = %pred.store.continue51
  %95 = or disjoint i32 %index.extracted, 23
  %96 = getelementptr inbounds i16, ptr %2, i32 %95
  %97 = or disjoint i16 %5, 23
  store i16 %97, ptr %96, align 2
  br label %pred.store.continue53

pred.store.continue53:                            ; preds = %pred.store.if52, %pred.store.continue51
  %98 = extractelement <64 x i1> %6, i64 23
  br i1 %98, label %pred.store.if54, label %pred.store.continue55

pred.store.if54:                                  ; preds = %pred.store.continue53
  %99 = or disjoint i32 %index.extracted, 24
  %100 = getelementptr inbounds i16, ptr %2, i32 %99
  %101 = or disjoint i16 %5, 24
  store i16 %101, ptr %100, align 2
  br label %pred.store.continue55

pred.store.continue55:                            ; preds = %pred.store.if54, %pred.store.continue53
  %102 = extractelement <64 x i1> %6, i64 24
  br i1 %102, label %pred.store.if56, label %pred.store.continue57

pred.store.if56:                                  ; preds = %pred.store.continue55
  %103 = or disjoint i32 %index.extracted, 25
  %104 = getelementptr inbounds i16, ptr %2, i32 %103
  %105 = or disjoint i16 %5, 25
  store i16 %105, ptr %104, align 2
  br label %pred.store.continue57

pred.store.continue57:                            ; preds = %pred.store.if56, %pred.store.continue55
  %106 = extractelement <64 x i1> %6, i64 25
  br i1 %106, label %pred.store.if58, label %pred.store.continue59

pred.store.if58:                                  ; preds = %pred.store.continue57
  %107 = or disjoint i32 %index.extracted, 26
  %108 = getelementptr inbounds i16, ptr %2, i32 %107
  %109 = or disjoint i16 %5, 26
  store i16 %109, ptr %108, align 2
  br label %pred.store.continue59

pred.store.continue59:                            ; preds = %pred.store.if58, %pred.store.continue57
  %110 = extractelement <64 x i1> %6, i64 26
  br i1 %110, label %pred.store.if60, label %pred.store.continue61

pred.store.if60:                                  ; preds = %pred.store.continue59
  %111 = or disjoint i32 %index.extracted, 27
  %112 = getelementptr inbounds i16, ptr %2, i32 %111
  %113 = or disjoint i16 %5, 27
  store i16 %113, ptr %112, align 2
  br label %pred.store.continue61

pred.store.continue61:                            ; preds = %pred.store.if60, %pred.store.continue59
  %114 = extractelement <64 x i1> %6, i64 27
  br i1 %114, label %pred.store.if62, label %pred.store.continue63

pred.store.if62:                                  ; preds = %pred.store.continue61
  %115 = or disjoint i32 %index.extracted, 28
  %116 = getelementptr inbounds i16, ptr %2, i32 %115
  %117 = or disjoint i16 %5, 28
  store i16 %117, ptr %116, align 2
  br label %pred.store.continue63

pred.store.continue63:                            ; preds = %pred.store.if62, %pred.store.continue61
  %118 = extractelement <64 x i1> %6, i64 28
  br i1 %118, label %pred.store.if64, label %pred.store.continue65

pred.store.if64:                                  ; preds = %pred.store.continue63
  %119 = or disjoint i32 %index.extracted, 29
  %120 = getelementptr inbounds i16, ptr %2, i32 %119
  %121 = or disjoint i16 %5, 29
  store i16 %121, ptr %120, align 2
  br label %pred.store.continue65

pred.store.continue65:                            ; preds = %pred.store.if64, %pred.store.continue63
  %122 = extractelement <64 x i1> %6, i64 29
  br i1 %122, label %pred.store.if66, label %pred.store.continue67

pred.store.if66:                                  ; preds = %pred.store.continue65
  %123 = or disjoint i32 %index.extracted, 30
  %124 = getelementptr inbounds i16, ptr %2, i32 %123
  %125 = or disjoint i16 %5, 30
  store i16 %125, ptr %124, align 2
  br label %pred.store.continue67

pred.store.continue67:                            ; preds = %pred.store.if66, %pred.store.continue65
  %126 = extractelement <64 x i1> %6, i64 30
  br i1 %126, label %pred.store.if68, label %pred.store.continue69

pred.store.if68:                                  ; preds = %pred.store.continue67
  %127 = or disjoint i32 %index.extracted, 31
  %128 = getelementptr inbounds i16, ptr %2, i32 %127
  %129 = or disjoint i16 %5, 31
  store i16 %129, ptr %128, align 2
  br label %pred.store.continue69

pred.store.continue69:                            ; preds = %pred.store.if68, %pred.store.continue67
  %130 = extractelement <64 x i1> %6, i64 31
  br i1 %130, label %pred.store.if70, label %pred.store.continue71

pred.store.if70:                                  ; preds = %pred.store.continue69
  %131 = or disjoint i32 %index.extracted, 32
  %132 = getelementptr inbounds i16, ptr %2, i32 %131
  %133 = or disjoint i16 %5, 32
  store i16 %133, ptr %132, align 2
  br label %pred.store.continue71

pred.store.continue71:                            ; preds = %pred.store.if70, %pred.store.continue69
  %134 = extractelement <64 x i1> %6, i64 32
  br i1 %134, label %pred.store.if72, label %pred.store.continue73

pred.store.if72:                                  ; preds = %pred.store.continue71
  %135 = or disjoint i32 %index.extracted, 33
  %136 = getelementptr inbounds i16, ptr %2, i32 %135
  %137 = or disjoint i16 %5, 33
  store i16 %137, ptr %136, align 2
  br label %pred.store.continue73

pred.store.continue73:                            ; preds = %pred.store.if72, %pred.store.continue71
  %138 = extractelement <64 x i1> %6, i64 33
  br i1 %138, label %pred.store.if74, label %pred.store.continue75

pred.store.if74:                                  ; preds = %pred.store.continue73
  %139 = or disjoint i32 %index.extracted, 34
  %140 = getelementptr inbounds i16, ptr %2, i32 %139
  %141 = or disjoint i16 %5, 34
  store i16 %141, ptr %140, align 2
  br label %pred.store.continue75

pred.store.continue75:                            ; preds = %pred.store.if74, %pred.store.continue73
  %142 = extractelement <64 x i1> %6, i64 34
  br i1 %142, label %pred.store.if76, label %pred.store.continue77

pred.store.if76:                                  ; preds = %pred.store.continue75
  %143 = or disjoint i32 %index.extracted, 35
  %144 = getelementptr inbounds i16, ptr %2, i32 %143
  %145 = or disjoint i16 %5, 35
  store i16 %145, ptr %144, align 2
  br label %pred.store.continue77

pred.store.continue77:                            ; preds = %pred.store.if76, %pred.store.continue75
  %146 = extractelement <64 x i1> %6, i64 35
  br i1 %146, label %pred.store.if78, label %pred.store.continue79

pred.store.if78:                                  ; preds = %pred.store.continue77
  %147 = or disjoint i32 %index.extracted, 36
  %148 = getelementptr inbounds i16, ptr %2, i32 %147
  %149 = or disjoint i16 %5, 36
  store i16 %149, ptr %148, align 2
  br label %pred.store.continue79

pred.store.continue79:                            ; preds = %pred.store.if78, %pred.store.continue77
  %150 = extractelement <64 x i1> %6, i64 36
  br i1 %150, label %pred.store.if80, label %pred.store.continue81

pred.store.if80:                                  ; preds = %pred.store.continue79
  %151 = or disjoint i32 %index.extracted, 37
  %152 = getelementptr inbounds i16, ptr %2, i32 %151
  %153 = or disjoint i16 %5, 37
  store i16 %153, ptr %152, align 2
  br label %pred.store.continue81

pred.store.continue81:                            ; preds = %pred.store.if80, %pred.store.continue79
  %154 = extractelement <64 x i1> %6, i64 37
  br i1 %154, label %pred.store.if82, label %pred.store.continue83

pred.store.if82:                                  ; preds = %pred.store.continue81
  %155 = or disjoint i32 %index.extracted, 38
  %156 = getelementptr inbounds i16, ptr %2, i32 %155
  %157 = or disjoint i16 %5, 38
  store i16 %157, ptr %156, align 2
  br label %pred.store.continue83

pred.store.continue83:                            ; preds = %pred.store.if82, %pred.store.continue81
  %158 = extractelement <64 x i1> %6, i64 38
  br i1 %158, label %pred.store.if84, label %pred.store.continue85

pred.store.if84:                                  ; preds = %pred.store.continue83
  %159 = or disjoint i32 %index.extracted, 39
  %160 = getelementptr inbounds i16, ptr %2, i32 %159
  %161 = or disjoint i16 %5, 39
  store i16 %161, ptr %160, align 2
  br label %pred.store.continue85

pred.store.continue85:                            ; preds = %pred.store.if84, %pred.store.continue83
  %162 = extractelement <64 x i1> %6, i64 39
  br i1 %162, label %pred.store.if86, label %pred.store.continue87

pred.store.if86:                                  ; preds = %pred.store.continue85
  %163 = or disjoint i32 %index.extracted, 40
  %164 = getelementptr inbounds i16, ptr %2, i32 %163
  %165 = or disjoint i16 %5, 40
  store i16 %165, ptr %164, align 2
  br label %pred.store.continue87

pred.store.continue87:                            ; preds = %pred.store.if86, %pred.store.continue85
  %166 = extractelement <64 x i1> %6, i64 40
  br i1 %166, label %pred.store.if88, label %pred.store.continue89

pred.store.if88:                                  ; preds = %pred.store.continue87
  %167 = or disjoint i32 %index.extracted, 41
  %168 = getelementptr inbounds i16, ptr %2, i32 %167
  %169 = or disjoint i16 %5, 41
  store i16 %169, ptr %168, align 2
  br label %pred.store.continue89

pred.store.continue89:                            ; preds = %pred.store.if88, %pred.store.continue87
  %170 = extractelement <64 x i1> %6, i64 41
  br i1 %170, label %pred.store.if90, label %pred.store.continue91

pred.store.if90:                                  ; preds = %pred.store.continue89
  %171 = or disjoint i32 %index.extracted, 42
  %172 = getelementptr inbounds i16, ptr %2, i32 %171
  %173 = or disjoint i16 %5, 42
  store i16 %173, ptr %172, align 2
  br label %pred.store.continue91

pred.store.continue91:                            ; preds = %pred.store.if90, %pred.store.continue89
  %174 = extractelement <64 x i1> %6, i64 42
  br i1 %174, label %pred.store.if92, label %pred.store.continue93

pred.store.if92:                                  ; preds = %pred.store.continue91
  %175 = or disjoint i32 %index.extracted, 43
  %176 = getelementptr inbounds i16, ptr %2, i32 %175
  %177 = or disjoint i16 %5, 43
  store i16 %177, ptr %176, align 2
  br label %pred.store.continue93

pred.store.continue93:                            ; preds = %pred.store.if92, %pred.store.continue91
  %178 = extractelement <64 x i1> %6, i64 43
  br i1 %178, label %pred.store.if94, label %pred.store.continue95

pred.store.if94:                                  ; preds = %pred.store.continue93
  %179 = or disjoint i32 %index.extracted, 44
  %180 = getelementptr inbounds i16, ptr %2, i32 %179
  %181 = or disjoint i16 %5, 44
  store i16 %181, ptr %180, align 2
  br label %pred.store.continue95

pred.store.continue95:                            ; preds = %pred.store.if94, %pred.store.continue93
  %182 = extractelement <64 x i1> %6, i64 44
  br i1 %182, label %pred.store.if96, label %pred.store.continue97

pred.store.if96:                                  ; preds = %pred.store.continue95
  %183 = or disjoint i32 %index.extracted, 45
  %184 = getelementptr inbounds i16, ptr %2, i32 %183
  %185 = or disjoint i16 %5, 45
  store i16 %185, ptr %184, align 2
  br label %pred.store.continue97

pred.store.continue97:                            ; preds = %pred.store.if96, %pred.store.continue95
  %186 = extractelement <64 x i1> %6, i64 45
  br i1 %186, label %pred.store.if98, label %pred.store.continue99

pred.store.if98:                                  ; preds = %pred.store.continue97
  %187 = or disjoint i32 %index.extracted, 46
  %188 = getelementptr inbounds i16, ptr %2, i32 %187
  %189 = or disjoint i16 %5, 46
  store i16 %189, ptr %188, align 2
  br label %pred.store.continue99

pred.store.continue99:                            ; preds = %pred.store.if98, %pred.store.continue97
  %190 = extractelement <64 x i1> %6, i64 46
  br i1 %190, label %pred.store.if100, label %pred.store.continue101

pred.store.if100:                                 ; preds = %pred.store.continue99
  %191 = or disjoint i32 %index.extracted, 47
  %192 = getelementptr inbounds i16, ptr %2, i32 %191
  %193 = or disjoint i16 %5, 47
  store i16 %193, ptr %192, align 2
  br label %pred.store.continue101

pred.store.continue101:                           ; preds = %pred.store.if100, %pred.store.continue99
  %194 = extractelement <64 x i1> %6, i64 47
  br i1 %194, label %pred.store.if102, label %pred.store.continue103

pred.store.if102:                                 ; preds = %pred.store.continue101
  %195 = or disjoint i32 %index.extracted, 48
  %196 = getelementptr inbounds i16, ptr %2, i32 %195
  %197 = or disjoint i16 %5, 48
  store i16 %197, ptr %196, align 2
  br label %pred.store.continue103

pred.store.continue103:                           ; preds = %pred.store.if102, %pred.store.continue101
  %198 = extractelement <64 x i1> %6, i64 48
  br i1 %198, label %pred.store.if104, label %pred.store.continue105

pred.store.if104:                                 ; preds = %pred.store.continue103
  %199 = or disjoint i32 %index.extracted, 49
  %200 = getelementptr inbounds i16, ptr %2, i32 %199
  %201 = or disjoint i16 %5, 49
  store i16 %201, ptr %200, align 2
  br label %pred.store.continue105

pred.store.continue105:                           ; preds = %pred.store.if104, %pred.store.continue103
  %202 = extractelement <64 x i1> %6, i64 49
  br i1 %202, label %pred.store.if106, label %pred.store.continue107

pred.store.if106:                                 ; preds = %pred.store.continue105
  %203 = or disjoint i32 %index.extracted, 50
  %204 = getelementptr inbounds i16, ptr %2, i32 %203
  %205 = or disjoint i16 %5, 50
  store i16 %205, ptr %204, align 2
  br label %pred.store.continue107

pred.store.continue107:                           ; preds = %pred.store.if106, %pred.store.continue105
  %206 = extractelement <64 x i1> %6, i64 50
  br i1 %206, label %pred.store.if108, label %pred.store.continue109

pred.store.if108:                                 ; preds = %pred.store.continue107
  %207 = or disjoint i32 %index.extracted, 51
  %208 = getelementptr inbounds i16, ptr %2, i32 %207
  %209 = or disjoint i16 %5, 51
  store i16 %209, ptr %208, align 2
  br label %pred.store.continue109

pred.store.continue109:                           ; preds = %pred.store.if108, %pred.store.continue107
  %210 = extractelement <64 x i1> %6, i64 51
  br i1 %210, label %pred.store.if110, label %pred.store.continue111

pred.store.if110:                                 ; preds = %pred.store.continue109
  %211 = or disjoint i32 %index.extracted, 52
  %212 = getelementptr inbounds i16, ptr %2, i32 %211
  %213 = or disjoint i16 %5, 52
  store i16 %213, ptr %212, align 2
  br label %pred.store.continue111

pred.store.continue111:                           ; preds = %pred.store.if110, %pred.store.continue109
  %214 = extractelement <64 x i1> %6, i64 52
  br i1 %214, label %pred.store.if112, label %pred.store.continue113

pred.store.if112:                                 ; preds = %pred.store.continue111
  %215 = or disjoint i32 %index.extracted, 53
  %216 = getelementptr inbounds i16, ptr %2, i32 %215
  %217 = or disjoint i16 %5, 53
  store i16 %217, ptr %216, align 2
  br label %pred.store.continue113

pred.store.continue113:                           ; preds = %pred.store.if112, %pred.store.continue111
  %218 = extractelement <64 x i1> %6, i64 53
  br i1 %218, label %pred.store.if114, label %pred.store.continue115

pred.store.if114:                                 ; preds = %pred.store.continue113
  %219 = or disjoint i32 %index.extracted, 54
  %220 = getelementptr inbounds i16, ptr %2, i32 %219
  %221 = or disjoint i16 %5, 54
  store i16 %221, ptr %220, align 2
  br label %pred.store.continue115

pred.store.continue115:                           ; preds = %pred.store.if114, %pred.store.continue113
  %222 = extractelement <64 x i1> %6, i64 54
  br i1 %222, label %pred.store.if116, label %pred.store.continue117

pred.store.if116:                                 ; preds = %pred.store.continue115
  %223 = or disjoint i32 %index.extracted, 55
  %224 = getelementptr inbounds i16, ptr %2, i32 %223
  %225 = or disjoint i16 %5, 55
  store i16 %225, ptr %224, align 2
  br label %pred.store.continue117

pred.store.continue117:                           ; preds = %pred.store.if116, %pred.store.continue115
  %226 = extractelement <64 x i1> %6, i64 55
  br i1 %226, label %pred.store.if118, label %pred.store.continue119

pred.store.if118:                                 ; preds = %pred.store.continue117
  %227 = or disjoint i32 %index.extracted, 56
  %228 = getelementptr inbounds i16, ptr %2, i32 %227
  %229 = or disjoint i16 %5, 56
  store i16 %229, ptr %228, align 2
  br label %pred.store.continue119

pred.store.continue119:                           ; preds = %pred.store.if118, %pred.store.continue117
  %230 = extractelement <64 x i1> %6, i64 56
  br i1 %230, label %pred.store.if120, label %pred.store.continue121

pred.store.if120:                                 ; preds = %pred.store.continue119
  %231 = or disjoint i32 %index.extracted, 57
  %232 = getelementptr inbounds i16, ptr %2, i32 %231
  %233 = or disjoint i16 %5, 57
  store i16 %233, ptr %232, align 2
  br label %pred.store.continue121

pred.store.continue121:                           ; preds = %pred.store.if120, %pred.store.continue119
  %234 = extractelement <64 x i1> %6, i64 57
  br i1 %234, label %pred.store.if122, label %pred.store.continue123

pred.store.if122:                                 ; preds = %pred.store.continue121
  %235 = or disjoint i32 %index.extracted, 58
  %236 = getelementptr inbounds i16, ptr %2, i32 %235
  %237 = or disjoint i16 %5, 58
  store i16 %237, ptr %236, align 2
  br label %pred.store.continue123

pred.store.continue123:                           ; preds = %pred.store.if122, %pred.store.continue121
  %238 = extractelement <64 x i1> %6, i64 58
  br i1 %238, label %pred.store.if124, label %pred.store.continue125

pred.store.if124:                                 ; preds = %pred.store.continue123
  %239 = or disjoint i32 %index.extracted, 59
  %240 = getelementptr inbounds i16, ptr %2, i32 %239
  %241 = or disjoint i16 %5, 59
  store i16 %241, ptr %240, align 2
  br label %pred.store.continue125

pred.store.continue125:                           ; preds = %pred.store.if124, %pred.store.continue123
  %242 = extractelement <64 x i1> %6, i64 59
  br i1 %242, label %pred.store.if126, label %pred.store.continue127

pred.store.if126:                                 ; preds = %pred.store.continue125
  %243 = or disjoint i32 %index.extracted, 60
  %244 = getelementptr inbounds i16, ptr %2, i32 %243
  %245 = or disjoint i16 %5, 60
  store i16 %245, ptr %244, align 2
  br label %pred.store.continue127

pred.store.continue127:                           ; preds = %pred.store.if126, %pred.store.continue125
  %246 = extractelement <64 x i1> %6, i64 60
  br i1 %246, label %pred.store.if128, label %pred.store.continue129

pred.store.if128:                                 ; preds = %pred.store.continue127
  %247 = or disjoint i32 %index.extracted, 61
  %248 = getelementptr inbounds i16, ptr %2, i32 %247
  %249 = or disjoint i16 %5, 61
  store i16 %249, ptr %248, align 2
  br label %pred.store.continue129

pred.store.continue129:                           ; preds = %pred.store.if128, %pred.store.continue127
  %250 = extractelement <64 x i1> %6, i64 61
  br i1 %250, label %pred.store.if130, label %pred.store.continue131

pred.store.if130:                                 ; preds = %pred.store.continue129
  %251 = or disjoint i32 %index.extracted, 62
  %252 = getelementptr inbounds i16, ptr %2, i32 %251
  %253 = or disjoint i16 %5, 62
  store i16 %253, ptr %252, align 2
  br label %pred.store.continue131

pred.store.continue131:                           ; preds = %pred.store.if130, %pred.store.continue129
  %254 = extractelement <64 x i1> %6, i64 62
  br i1 %254, label %pred.store.if132, label %pred.store.continue133

pred.store.if132:                                 ; preds = %pred.store.continue131
  %255 = or disjoint i32 %index.extracted, 63
  %256 = getelementptr inbounds i16, ptr %2, i32 %255
  %257 = or disjoint i16 %5, 63
  store i16 %257, ptr %256, align 2
  br label %pred.store.continue133

pred.store.continue133:                           ; preds = %pred.store.if132, %pred.store.continue131
  %258 = extractelement <64 x i1> %6, i64 63
  br i1 %258, label %pred.store.if134, label %pred.store.continue135

pred.store.if134:                                 ; preds = %pred.store.continue133
  %gep = getelementptr i16, ptr %3, i32 %index.extracted
  %259 = add i16 %5, 64
  store i16 %259, ptr %gep, align 2
  br label %pred.store.continue135

pred.store.continue135:                           ; preds = %pred.store.if134, %pred.store.continue133
  %index.next = add i32 %index.extracted, 64
  %260 = icmp eq i32 %index.next, %4
  br i1 %260, label %exit, label %vector.body.extracted_entry

exit:                                             ; preds = %pred.store.continue135
  ret void
}



