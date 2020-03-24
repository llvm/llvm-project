; RUN: opt < %s -tapir2target -tapir-target=cilk -debug-abi-calls -cilk-use-arg-struct -S | FileCheck %s
; RUN: opt < %s -passes=tapir2target -tapir-target=cilk -debug-abi-calls -cilk-use-arg-struct -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, %struct._IO_codecvt*, %struct._IO_wide_data*, %struct._IO_FILE*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type opaque
%struct._IO_codecvt = type opaque
%struct._IO_wide_data = type opaque
%struct.timeval = type { i64, i64 }
%struct.timezone = type { i32, i32 }

@stderr = external dso_local local_unnamed_addr global %struct._IO_FILE*, align 8
@.str = private unnamed_addr constant [69 x i8] c"\0AUsage: cilksort [<cilk-options>] [-n size] [-c] [-benchmark] [-h]\0A\0A\00", align 1
@.str.1 = private unnamed_addr constant [69 x i8] c"Cilksort is a parallel sorting algorithm, donned \22Multisort\22, which\0A\00", align 1
@.str.2 = private unnamed_addr constant [70 x i8] c"is a variant of ordinary mergesort.  Multisort begins by dividing an\0A\00", align 1
@.str.3 = private unnamed_addr constant [70 x i8] c"array of elements in half and sorting each half.  It then merges the\0A\00", align 1
@.str.4 = private unnamed_addr constant [71 x i8] c"two sorted halves back together, but in a divide-and-conquer approach\0A\00", align 1
@.str.5 = private unnamed_addr constant [38 x i8] c"rather than the usual serial merge.\0A\0A\00", align 1
@.str.6 = private unnamed_addr constant [3 x i8] c"-n\00", align 1
@.str.7 = private unnamed_addr constant [3 x i8] c"-c\00", align 1
@.str.8 = private unnamed_addr constant [11 x i8] c"-benchmark\00", align 1
@.str.9 = private unnamed_addr constant [3 x i8] c"-h\00", align 1
@specifiers = dso_local global [5 x i8*] [i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.6, i32 0, i32 0), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.7, i32 0, i32 0), i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.8, i32 0, i32 0), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.9, i32 0, i32 0), i8* null], align 16
@opt_types = dso_local global [5 x i32] [i32 3, i32 4, i32 6, i32 4, i32 0], align 16
@.str.10 = private unnamed_addr constant [4 x i8] c"%f\0A\00", align 1
@.str.12 = private unnamed_addr constant [17 x i8] c"SORTING FAILURE!\00", align 1
@.str.13 = private unnamed_addr constant [20 x i8] c"Sorting successful.\00", align 1
@.str.14 = private unnamed_addr constant [25 x i8] c"\0ACilk Example: cilksort\0A\00", align 1
@.str.15 = private unnamed_addr constant [36 x i8] c"options: number of elements = %ld\0A\0A\00", align 1
@rand_nxt = internal unnamed_addr global i64 0, align 8
@str = private unnamed_addr constant [22 x i8] c"Now check result ... \00", align 1

; Function Attrs: argmemonly norecurse nounwind readonly uwtable
define dso_local i64 @todval(%struct.timeval* nocapture readonly %tp) local_unnamed_addr #0 {
entry:
  %tv_sec = getelementptr inbounds %struct.timeval, %struct.timeval* %tp, i64 0, i32 0
  %0 = load i64, i64* %tv_sec, align 8, !tbaa !2
  %mul1 = mul i64 %0, 1000000
  %tv_usec = getelementptr inbounds %struct.timeval, %struct.timeval* %tp, i64 0, i32 1
  %1 = load i64, i64* %tv_usec, align 8, !tbaa !7
  %add = add nsw i64 %mul1, %1
  ret i64 %add
}

; Function Attrs: argmemonly nounwind uwtable
define dso_local void @seqquick(i64* %low, i64* %high) local_unnamed_addr #1 {
entry:
  %sub.ptr.lhs.cast = ptrtoint i64* %high to i64
  %sub.ptr.rhs.cast6 = ptrtoint i64* %low to i64
  %sub.ptr.sub7 = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast6
  %cmp8 = icmp sgt i64 %sub.ptr.sub7, 152
  br i1 %cmp8, label %while.body, label %while.end

while.body:                                       ; preds = %entry, %while.body
  %low.addr.09 = phi i64* [ %add.ptr, %while.body ], [ %low, %entry ]
  %call = tail call fastcc i64* @seqpart(i64* %low.addr.09, i64* %high)
  tail call void @seqquick(i64* %low.addr.09, i64* %call)
  %add.ptr = getelementptr inbounds i64, i64* %call, i64 1
  %sub.ptr.rhs.cast = ptrtoint i64* %add.ptr to i64
  %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  %cmp = icmp sgt i64 %sub.ptr.sub, 152
  br i1 %cmp, label %while.body, label %while.end

while.end:                                        ; preds = %while.body, %entry
  %low.addr.0.lcssa = phi i64* [ %low, %entry ], [ %add.ptr, %while.body ]
  tail call fastcc void @insertion_sort(i64* %low.addr.0.lcssa, i64* %high)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #2

; Function Attrs: argmemonly norecurse nounwind uwtable
define internal fastcc i64* @seqpart(i64* %low, i64* %high) unnamed_addr #3 {
entry:
  %call = tail call fastcc i64 @choose_pivot(i64* %low, i64* %high)
  br label %while.cond

while.cond:                                       ; preds = %if.end, %entry
  %curr_low.0 = phi i64* [ %low, %entry ], [ %incdec.ptr10, %if.end ]
  %curr_high.0 = phi i64* [ %high, %entry ], [ %incdec.ptr9, %if.end ]
  br label %while.cond1

while.cond1:                                      ; preds = %while.cond1, %while.cond
  %curr_high.1 = phi i64* [ %curr_high.0, %while.cond ], [ %incdec.ptr, %while.cond1 ]
  %0 = load i64, i64* %curr_high.1, align 8, !tbaa !8
  %cmp = icmp sgt i64 %0, %call
  %incdec.ptr = getelementptr inbounds i64, i64* %curr_high.1, i64 -1
  br i1 %cmp, label %while.cond1, label %while.cond3

while.cond3:                                      ; preds = %while.cond1, %while.cond3
  %curr_low.1 = phi i64* [ %incdec.ptr6, %while.cond3 ], [ %curr_low.0, %while.cond1 ]
  %1 = load i64, i64* %curr_low.1, align 8, !tbaa !8
  %cmp4 = icmp slt i64 %1, %call
  %incdec.ptr6 = getelementptr inbounds i64, i64* %curr_low.1, i64 1
  br i1 %cmp4, label %while.cond3, label %while.end7

while.end7:                                       ; preds = %while.cond3
  %cmp8 = icmp ult i64* %curr_low.1, %curr_high.1
  br i1 %cmp8, label %if.end, label %while.end11

if.end:                                           ; preds = %while.end7
  %incdec.ptr9 = getelementptr inbounds i64, i64* %curr_high.1, i64 -1
  store i64 %1, i64* %curr_high.1, align 8, !tbaa !8
  %incdec.ptr10 = getelementptr inbounds i64, i64* %curr_low.1, i64 1
  store i64 %0, i64* %curr_low.1, align 8, !tbaa !8
  br label %while.cond

while.end11:                                      ; preds = %while.end7
  %cmp12 = icmp ult i64* %curr_high.1, %high
  %add.ptr = getelementptr inbounds i64, i64* %curr_high.1, i64 -1
  %retval.0 = select i1 %cmp12, i64* %curr_high.1, i64* %add.ptr
  ret i64* %retval.0
}

; Function Attrs: argmemonly norecurse nounwind uwtable
define internal fastcc void @insertion_sort(i64* %low, i64* readnone %high) unnamed_addr #3 {
entry:
  %q.028 = getelementptr inbounds i64, i64* %low, i64 1
  %cmp29 = icmp ugt i64* %q.028, %high
  br i1 %cmp29, label %for.end11, label %for.body

for.body:                                         ; preds = %entry, %for.end
  %q.031 = phi i64* [ %q.0, %for.end ], [ %q.028, %entry ]
  %low.pn30 = phi i64* [ %q.031, %for.end ], [ %low, %entry ]
  %0 = load i64, i64* %q.031, align 8, !tbaa !8
  %cmp325 = icmp ult i64* %low.pn30, %low
  br i1 %cmp325, label %for.end, label %land.rhs

land.rhs:                                         ; preds = %for.body, %for.body6
  %p.026 = phi i64* [ %incdec.ptr, %for.body6 ], [ %low.pn30, %for.body ]
  %1 = load i64, i64* %p.026, align 8, !tbaa !8
  %cmp5 = icmp sgt i64 %1, %0
  br i1 %cmp5, label %for.body6, label %for.end

for.body6:                                        ; preds = %land.rhs
  %arrayidx7 = getelementptr inbounds i64, i64* %p.026, i64 1
  store i64 %1, i64* %arrayidx7, align 8, !tbaa !8
  %incdec.ptr = getelementptr inbounds i64, i64* %p.026, i64 -1
  %cmp3 = icmp ult i64* %incdec.ptr, %low
  br i1 %cmp3, label %for.end, label %land.rhs

for.end:                                          ; preds = %land.rhs, %for.body6, %for.body
  %p.0.lcssa = phi i64* [ %low.pn30, %for.body ], [ %incdec.ptr, %for.body6 ], [ %p.026, %land.rhs ]
  %arrayidx8 = getelementptr inbounds i64, i64* %p.0.lcssa, i64 1
  store i64 %0, i64* %arrayidx8, align 8, !tbaa !8
  %q.0 = getelementptr inbounds i64, i64* %q.031, i64 1
  %cmp = icmp ugt i64* %q.0, %high
  br i1 %cmp, label %for.end11, label %for.body

for.end11:                                        ; preds = %for.end, %entry
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #2

; Function Attrs: argmemonly nounwind uwtable
define dso_local void @seqmerge(i64* %low1, i64* %high1, i64* %low2, i64* %high2, i64* nocapture %lowdest) local_unnamed_addr #1 {
entry:
  %cmp = icmp ult i64* %low1, %high1
  %cmp1 = icmp ult i64* %low2, %high2
  %or.cond = and i1 %cmp, %cmp1
  br i1 %or.cond, label %if.then, label %if.end13

if.then:                                          ; preds = %entry
  %0 = load i64, i64* %low1, align 8, !tbaa !8
  %1 = load i64, i64* %low2, align 8, !tbaa !8
  br label %for.cond.outer

for.cond.outer:                                   ; preds = %if.else, %if.then
  %low2.addr.0.ph = phi i64* [ %incdec.ptr8, %if.else ], [ %low2, %if.then ]
  %low1.addr.0.ph = phi i64* [ %low1.addr.0, %if.else ], [ %low1, %if.then ]
  %lowdest.addr.0.ph = phi i64* [ %incdec.ptr, %if.else ], [ %lowdest, %if.then ]
  %a1.0.ph = phi i64 [ %a1.0, %if.else ], [ %0, %if.then ]
  %a2.0.ph = phi i64 [ %3, %if.else ], [ %1, %if.then ]
  br label %for.cond

for.cond:                                         ; preds = %for.cond.outer, %if.then3
  %low1.addr.0 = phi i64* [ %incdec.ptr4, %if.then3 ], [ %low1.addr.0.ph, %for.cond.outer ]
  %lowdest.addr.0 = phi i64* [ %incdec.ptr, %if.then3 ], [ %lowdest.addr.0.ph, %for.cond.outer ]
  %a1.0 = phi i64 [ %2, %if.then3 ], [ %a1.0.ph, %for.cond.outer ]
  %cmp2 = icmp slt i64 %a1.0, %a2.0.ph
  %incdec.ptr = getelementptr inbounds i64, i64* %lowdest.addr.0, i64 1
  br i1 %cmp2, label %if.then3, label %if.else

if.then3:                                         ; preds = %for.cond
  store i64 %a1.0, i64* %lowdest.addr.0, align 8, !tbaa !8
  %incdec.ptr4 = getelementptr inbounds i64, i64* %low1.addr.0, i64 1
  %2 = load i64, i64* %incdec.ptr4, align 8, !tbaa !8
  %cmp5 = icmp ult i64* %incdec.ptr4, %high1
  br i1 %cmp5, label %for.cond, label %if.end13

if.else:                                          ; preds = %for.cond
  store i64 %a2.0.ph, i64* %lowdest.addr.0, align 8, !tbaa !8
  %incdec.ptr8 = getelementptr inbounds i64, i64* %low2.addr.0.ph, i64 1
  %3 = load i64, i64* %incdec.ptr8, align 8, !tbaa !8
  %cmp9 = icmp ult i64* %incdec.ptr8, %high2
  br i1 %cmp9, label %for.cond.outer, label %if.end13

if.end13:                                         ; preds = %if.then3, %if.else, %entry
  %low2.addr.2 = phi i64* [ %low2, %entry ], [ %low2.addr.0.ph, %if.then3 ], [ %incdec.ptr8, %if.else ]
  %low1.addr.2 = phi i64* [ %low1, %entry ], [ %incdec.ptr4, %if.then3 ], [ %low1.addr.0, %if.else ]
  %lowdest.addr.2 = phi i64* [ %lowdest, %entry ], [ %incdec.ptr, %if.else ], [ %incdec.ptr, %if.then3 ]
  %cmp14 = icmp ugt i64* %low1.addr.2, %high1
  %cmp16 = icmp ugt i64* %low2.addr.2, %high2
  %or.cond87 = or i1 %cmp16, %cmp14
  br i1 %or.cond87, label %if.end34, label %if.then17

if.then17:                                        ; preds = %if.end13
  %4 = load i64, i64* %low1.addr.2, align 8, !tbaa !8
  %5 = load i64, i64* %low2.addr.2, align 8, !tbaa !8
  br label %for.cond18.outer

for.cond18.outer:                                 ; preds = %if.end31, %if.then17
  %low2.addr.3.ph = phi i64* [ %incdec.ptr28, %if.end31 ], [ %low2.addr.2, %if.then17 ]
  %low1.addr.3.ph = phi i64* [ %low1.addr.3, %if.end31 ], [ %low1.addr.2, %if.then17 ]
  %lowdest.addr.3.ph = phi i64* [ %incdec.ptr21, %if.end31 ], [ %lowdest.addr.2, %if.then17 ]
  %a1.2.ph = phi i64 [ %a1.2, %if.end31 ], [ %4, %if.then17 ]
  %a2.2.ph = phi i64 [ %7, %if.end31 ], [ %5, %if.then17 ]
  br label %for.cond18

for.cond18:                                       ; preds = %for.cond18.outer, %if.end25
  %low1.addr.3 = phi i64* [ %incdec.ptr22, %if.end25 ], [ %low1.addr.3.ph, %for.cond18.outer ]
  %lowdest.addr.3 = phi i64* [ %incdec.ptr21, %if.end25 ], [ %lowdest.addr.3.ph, %for.cond18.outer ]
  %a1.2 = phi i64 [ %6, %if.end25 ], [ %a1.2.ph, %for.cond18.outer ]
  %cmp19 = icmp slt i64 %a1.2, %a2.2.ph
  %incdec.ptr21 = getelementptr inbounds i64, i64* %lowdest.addr.3, i64 1
  br i1 %cmp19, label %if.then20, label %if.else26

if.then20:                                        ; preds = %for.cond18
  store i64 %a1.2, i64* %lowdest.addr.3, align 8, !tbaa !8
  %incdec.ptr22 = getelementptr inbounds i64, i64* %low1.addr.3, i64 1
  %cmp23 = icmp ugt i64* %incdec.ptr22, %high1
  br i1 %cmp23, label %if.end34, label %if.end25

if.end25:                                         ; preds = %if.then20
  %6 = load i64, i64* %incdec.ptr22, align 8, !tbaa !8
  br label %for.cond18

if.else26:                                        ; preds = %for.cond18
  store i64 %a2.2.ph, i64* %lowdest.addr.3, align 8, !tbaa !8
  %incdec.ptr28 = getelementptr inbounds i64, i64* %low2.addr.3.ph, i64 1
  %cmp29 = icmp ugt i64* %incdec.ptr28, %high2
  br i1 %cmp29, label %if.end34, label %if.end31

if.end31:                                         ; preds = %if.else26
  %7 = load i64, i64* %incdec.ptr28, align 8, !tbaa !8
  br label %for.cond18.outer

if.end34:                                         ; preds = %if.then20, %if.else26, %if.end13
  %low2.addr.5 = phi i64* [ %low2.addr.2, %if.end13 ], [ %low2.addr.3.ph, %if.then20 ], [ %incdec.ptr28, %if.else26 ]
  %low1.addr.5 = phi i64* [ %low1.addr.2, %if.end13 ], [ %incdec.ptr22, %if.then20 ], [ %low1.addr.3, %if.else26 ]
  %lowdest.addr.5 = phi i64* [ %lowdest.addr.2, %if.end13 ], [ %incdec.ptr21, %if.else26 ], [ %incdec.ptr21, %if.then20 ]
  %cmp35 = icmp ugt i64* %low1.addr.5, %high1
  %8 = bitcast i64* %lowdest.addr.5 to i8*
  br i1 %cmp35, label %if.then36, label %if.else37

if.then36:                                        ; preds = %if.end34
  %9 = bitcast i64* %low2.addr.5 to i8*
  %sub.ptr.lhs.cast = ptrtoint i64* %high2 to i64
  %sub.ptr.rhs.cast = ptrtoint i64* %low2.addr.5 to i64
  %sub.ptr.sub = add i64 %sub.ptr.lhs.cast, 8
  %10 = sub i64 %sub.ptr.sub, %sub.ptr.rhs.cast
  %mul = and i64 %10, -8
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %8, i8* align 8 %9, i64 %mul, i1 false)
  br label %if.end44

if.else37:                                        ; preds = %if.end34
  %11 = bitcast i64* %low1.addr.5 to i8*
  %sub.ptr.lhs.cast38 = ptrtoint i64* %high1 to i64
  %sub.ptr.rhs.cast39 = ptrtoint i64* %low1.addr.5 to i64
  %sub.ptr.sub40 = add i64 %sub.ptr.lhs.cast38, 8
  %12 = sub i64 %sub.ptr.sub40, %sub.ptr.rhs.cast39
  %mul43 = and i64 %12, -8
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %8, i8* align 8 %11, i64 %mul43, i1 false)
  br label %if.end44

if.end44:                                         ; preds = %if.else37, %if.then36
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #2

; Function Attrs: argmemonly norecurse nounwind readonly uwtable
define dso_local i64* @binsplit(i64 %val, i64* %low, i64* %high) local_unnamed_addr #0 {
entry:
  %cmp15 = icmp eq i64* %low, %high
  br i1 %cmp15, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %high.addr.017 = phi i64* [ %high.addr.1, %while.body ], [ %high, %entry ]
  %low.addr.016 = phi i64* [ %low.addr.1, %while.body ], [ %low, %entry ]
  %sub.ptr.lhs.cast = ptrtoint i64* %high.addr.017 to i64
  %sub.ptr.rhs.cast = ptrtoint i64* %low.addr.016 to i64
  %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  %sub.ptr.div = ashr exact i64 %sub.ptr.sub, 3
  %add = add nsw i64 %sub.ptr.div, 1
  %shr = ashr i64 %add, 1
  %add.ptr = getelementptr inbounds i64, i64* %low.addr.016, i64 %shr
  %0 = load i64, i64* %add.ptr, align 8, !tbaa !8
  %cmp1 = icmp slt i64 %0, %val
  %add.ptr2 = getelementptr inbounds i64, i64* %add.ptr, i64 -1
  %low.addr.1 = select i1 %cmp1, i64* %add.ptr, i64* %low.addr.016
  %high.addr.1 = select i1 %cmp1, i64* %high.addr.017, i64* %add.ptr2
  %cmp = icmp eq i64* %low.addr.1, %high.addr.1
  br i1 %cmp, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  %low.addr.0.lcssa = phi i64* [ %low, %entry ], [ %low.addr.1, %while.body ]
  %1 = load i64, i64* %low.addr.0.lcssa, align 8, !tbaa !8
  %cmp3 = icmp sgt i64 %1, %val
  %add.ptr5 = getelementptr inbounds i64, i64* %low.addr.0.lcssa, i64 -1
  %retval.0 = select i1 %cmp3, i64* %add.ptr5, i64* %low.addr.0.lcssa
  ret i64* %retval.0
}

; Function Attrs: argmemonly nounwind uwtable
define dso_local void @cilkmerge(i64* %low1, i64* %high1, i64* %low2, i64* %high2, i64* %lowdest) local_unnamed_addr #1 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %sub.ptr.lhs.cast94 = ptrtoint i64* %high2 to i64
  %sub.ptr.rhs.cast95 = ptrtoint i64* %low2 to i64
  %sub.ptr.sub96 = sub i64 %sub.ptr.lhs.cast94, %sub.ptr.rhs.cast95
  %sub.ptr.lhs.cast197 = ptrtoint i64* %high1 to i64
  %sub.ptr.rhs.cast298 = ptrtoint i64* %low1 to i64
  %sub.ptr.sub399 = sub i64 %sub.ptr.lhs.cast197, %sub.ptr.rhs.cast298
  %cmp100 = icmp sgt i64 %sub.ptr.sub96, %sub.ptr.sub399
  %spec.select101 = select i1 %cmp100, i64* %high1, i64* %high2
  %spec.select86102 = select i1 %cmp100, i64* %low1, i64* %low2
  %spec.select87103 = select i1 %cmp100, i64* %high2, i64* %high1
  %spec.select88104 = select i1 %cmp100, i64* %low2, i64* %low1
  %cmp6105 = icmp ult i64* %spec.select87103, %spec.select88104
  br i1 %cmp6105, label %if.then7, label %if.end12

if.then7:                                         ; preds = %det.cont, %entry
  %lowdest.tr.lcssa = phi i64* [ %lowdest, %entry ], [ %add.ptr39, %det.cont ]
  %spec.select.lcssa = phi i64* [ %spec.select101, %entry ], [ %spec.select, %det.cont ]
  %spec.select86.lcssa = phi i64* [ %spec.select86102, %entry ], [ %spec.select86, %det.cont ]
  %0 = bitcast i64* %lowdest.tr.lcssa to i8*
  %1 = bitcast i64* %spec.select86.lcssa to i8*
  %sub.ptr.lhs.cast8 = ptrtoint i64* %spec.select.lcssa to i64
  %sub.ptr.rhs.cast9 = ptrtoint i64* %spec.select86.lcssa to i64
  %sub.ptr.sub10 = sub i64 %sub.ptr.lhs.cast8, %sub.ptr.rhs.cast9
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %0, i8* align 8 %1, i64 %sub.ptr.sub10, i1 false)
  br label %cleanup

if.end12:                                         ; preds = %entry, %det.cont
  %spec.select88110 = phi i64* [ %spec.select88, %det.cont ], [ %spec.select88104, %entry ]
  %spec.select87109 = phi i64* [ %spec.select87, %det.cont ], [ %spec.select87103, %entry ]
  %spec.select86108 = phi i64* [ %spec.select86, %det.cont ], [ %spec.select86102, %entry ]
  %spec.select107 = phi i64* [ %spec.select, %det.cont ], [ %spec.select101, %entry ]
  %lowdest.tr106 = phi i64* [ %add.ptr39, %det.cont ], [ %lowdest, %entry ]
  %sub.ptr.lhs.cast13 = ptrtoint i64* %spec.select107 to i64
  %sub.ptr.rhs.cast14 = ptrtoint i64* %spec.select86108 to i64
  %sub.ptr.sub15 = sub i64 %sub.ptr.lhs.cast13, %sub.ptr.rhs.cast14
  %cmp17 = icmp slt i64 %sub.ptr.sub15, 16384
  br i1 %cmp17, label %if.then18, label %if.end19

if.then18:                                        ; preds = %if.end12
  tail call void @seqmerge(i64* %spec.select88110, i64* %spec.select87109, i64* %spec.select86108, i64* %spec.select107, i64* %lowdest.tr106)
  br label %cleanup

if.end19:                                         ; preds = %if.end12
  %sub.ptr.lhs.cast20 = ptrtoint i64* %spec.select87109 to i64
  %sub.ptr.rhs.cast21 = ptrtoint i64* %spec.select88110 to i64
  %sub.ptr.sub22 = sub i64 %sub.ptr.lhs.cast20, %sub.ptr.rhs.cast21
  %sub.ptr.div23 = ashr exact i64 %sub.ptr.sub22, 3
  %add = add nsw i64 %sub.ptr.div23, 1
  %div = sdiv i64 %add, 2
  %add.ptr = getelementptr inbounds i64, i64* %spec.select88110, i64 %div
  %2 = load i64, i64* %add.ptr, align 8, !tbaa !8
  %call = tail call i64* @binsplit(i64 %2, i64* %spec.select86108, i64* %spec.select107)
  %add.ptr28 = getelementptr inbounds i64, i64* %call, i64 %div
  %sub.ptr.lhs.cast29 = ptrtoint i64* %add.ptr28 to i64
  %sub.ptr.sub31 = sub i64 %sub.ptr.lhs.cast29, %sub.ptr.rhs.cast14
  %sub.ptr.div32 = ashr exact i64 %sub.ptr.sub31, 3
  %add.ptr33 = getelementptr inbounds i64, i64* %lowdest.tr106, i64 %sub.ptr.div32
  %add.ptr34 = getelementptr inbounds i64, i64* %add.ptr33, i64 1
  store i64 %2, i64* %add.ptr34, align 8, !tbaa !8
  detach within %syncreg, label %det.achd, label %det.cont

det.achd:                                         ; preds = %if.end19
  %add.ptr35 = getelementptr inbounds i64, i64* %add.ptr, i64 -1
  tail call void @cilkmerge(i64* nonnull %spec.select88110, i64* nonnull %add.ptr35, i64* %spec.select86108, i64* %call, i64* nonnull %lowdest.tr106)
  reattach within %syncreg, label %det.cont

det.cont:                                         ; preds = %det.achd, %if.end19
  %add.ptr36 = getelementptr inbounds i64, i64* %add.ptr, i64 1
  %add.ptr37 = getelementptr inbounds i64, i64* %call, i64 1
  %add.ptr39 = getelementptr inbounds i64, i64* %add.ptr33, i64 2
  %sub.ptr.lhs.cast = ptrtoint i64* %spec.select107 to i64
  %sub.ptr.rhs.cast = ptrtoint i64* %add.ptr37 to i64
  %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  %sub.ptr.lhs.cast1 = ptrtoint i64* %spec.select87109 to i64
  %sub.ptr.rhs.cast2 = ptrtoint i64* %add.ptr36 to i64
  %sub.ptr.sub3 = sub i64 %sub.ptr.lhs.cast1, %sub.ptr.rhs.cast2
  %cmp = icmp sgt i64 %sub.ptr.sub, %sub.ptr.sub3
  %spec.select = select i1 %cmp, i64* %spec.select87109, i64* %spec.select107
  %spec.select86 = select i1 %cmp, i64* %add.ptr36, i64* %add.ptr37
  %spec.select87 = select i1 %cmp, i64* %spec.select107, i64* %spec.select87109
  %spec.select88 = select i1 %cmp, i64* %add.ptr37, i64* %add.ptr36
  %cmp6 = icmp ult i64* %spec.select87, %spec.select88
  br i1 %cmp6, label %if.then7, label %if.end12

cleanup:                                          ; preds = %if.then18, %if.then7
  sync within %syncreg, label %cleanup.split

cleanup.split:                                    ; preds = %cleanup
  ret void
}

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #2

; Function Attrs: argmemonly nounwind uwtable
define dso_local void @cilksort(i64* %low, i64* %tmp, i64 %size) local_unnamed_addr #1 {
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %div = sdiv i64 %size, 4
  %cmp = icmp slt i64 %size, 2048
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %add.ptr = getelementptr inbounds i64, i64* %low, i64 %size
  %add.ptr1 = getelementptr inbounds i64, i64* %add.ptr, i64 -1
  tail call void @seqquick(i64* %low, i64* nonnull %add.ptr1)
  br label %cleanup

if.end:                                           ; preds = %entry
  %add.ptr2 = getelementptr inbounds i64, i64* %low, i64 %div
  %add.ptr3 = getelementptr inbounds i64, i64* %tmp, i64 %div
  %add.ptr4 = getelementptr inbounds i64, i64* %add.ptr2, i64 %div
  %add.ptr5 = getelementptr inbounds i64, i64* %add.ptr3, i64 %div
  %add.ptr6 = getelementptr inbounds i64, i64* %add.ptr4, i64 %div
  %add.ptr7 = getelementptr inbounds i64, i64* %add.ptr5, i64 %div
  detach within %syncreg, label %det.achd, label %det.cont

det.achd:                                         ; preds = %if.end
  tail call void @cilksort(i64* %low, i64* %tmp, i64 %div)
  reattach within %syncreg, label %det.cont


det.cont:                                         ; preds = %det.achd, %if.end
  detach within %syncreg, label %det.achd8, label %det.cont9

det.achd8:                                        ; preds = %det.cont
  tail call void @cilksort(i64* %add.ptr2, i64* %add.ptr3, i64 %div)
  reattach within %syncreg, label %det.cont9

det.cont9:                                        ; preds = %det.achd8, %det.cont
  detach within %syncreg, label %det.achd10, label %det.cont11

det.achd10:                                       ; preds = %det.cont9
  tail call void @cilksort(i64* %add.ptr4, i64* %add.ptr5, i64 %div)
  reattach within %syncreg, label %det.cont11

det.cont11:                                       ; preds = %det.achd10, %det.cont9
  %0 = mul i64 %div, -3
  %sub = add i64 %0, %size
  tail call void @cilksort(i64* %add.ptr6, i64* %add.ptr7, i64 %sub)
  sync within %syncreg, label %sync.continue

sync.continue:                                    ; preds = %det.cont11
  detach within %syncreg, label %det.achd16, label %det.cont17

det.achd16:                                       ; preds = %sync.continue
  %add.ptr15 = getelementptr inbounds i64, i64* %add.ptr4, i64 -1
  %add.ptr13 = getelementptr inbounds i64, i64* %add.ptr2, i64 -1
  tail call void @cilkmerge(i64* %low, i64* nonnull %add.ptr13, i64* %add.ptr2, i64* nonnull %add.ptr15, i64* %tmp)
  reattach within %syncreg, label %det.cont17

det.cont17:                                       ; preds = %det.achd16, %sync.continue
  %add.ptr19 = getelementptr inbounds i64, i64* %add.ptr6, i64 -1
  %add.ptr20 = getelementptr inbounds i64, i64* %low, i64 %size
  %add.ptr21 = getelementptr inbounds i64, i64* %add.ptr20, i64 -1
  tail call void @cilkmerge(i64* %add.ptr4, i64* nonnull %add.ptr19, i64* %add.ptr6, i64* nonnull %add.ptr21, i64* %add.ptr5)
  sync within %syncreg, label %sync.continue22

sync.continue22:                                  ; preds = %det.cont17
  %add.ptr23 = getelementptr inbounds i64, i64* %add.ptr5, i64 -1
  %add.ptr24 = getelementptr inbounds i64, i64* %tmp, i64 %size
  %add.ptr25 = getelementptr inbounds i64, i64* %add.ptr24, i64 -1
  tail call void @cilkmerge(i64* %tmp, i64* nonnull %add.ptr23, i64* %add.ptr5, i64* nonnull %add.ptr25, i64* %low)
  br label %cleanup

cleanup:                                          ; preds = %sync.continue22, %if.then
  ret void
}

; CHECK: define {{.*}}void @cilksort.outline_det.achd16.otd1({{.*}}) {{.*}}#[[DETACHD16ATTR:[0-9]+]] {

; CHECK: attributes #[[DETACHD16ATTR]] = {
; CHECK-NOT: argmemonly
; CHECK-NOT: inaccessiblememonly
; CHECK-NOT: inaccessiblemem_or_argmemonly
; CHECK: }

; Function Attrs: norecurse nounwind uwtable
define dso_local void @scramble_array(i64* nocapture %arr, i64 %size) local_unnamed_addr #4 {
entry:
  %cmp16 = icmp eq i64 %size, 0
  br i1 %cmp16, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.017 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %call = tail call fastcc i64 @my_rand()
  %rem = urem i64 %call, %size
  %arrayidx = getelementptr inbounds i64, i64* %arr, i64 %i.017
  %0 = load i64, i64* %arrayidx, align 8, !tbaa !8
  %arrayidx1 = getelementptr inbounds i64, i64* %arr, i64 %rem
  %1 = load i64, i64* %arrayidx1, align 8, !tbaa !8
  store i64 %1, i64* %arrayidx, align 8, !tbaa !8
  store i64 %0, i64* %arrayidx1, align 8, !tbaa !8
  %inc = add nuw i64 %i.017, 1
  %exitcond = icmp eq i64 %inc, %size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

; Function Attrs: inlinehint norecurse nounwind uwtable
define internal fastcc i64 @my_rand() unnamed_addr #5 {
entry:
  %0 = load i64, i64* @rand_nxt, align 8, !tbaa !8
  %mul = mul i64 %0, 1103515245
  %add = add i64 %mul, 12345
  store i64 %add, i64* @rand_nxt, align 8, !tbaa !8
  ret i64 %add
}

; Function Attrs: norecurse nounwind uwtable
define dso_local void @fill_array(i64* nocapture %arr, i64 %size) local_unnamed_addr #4 {
entry:
  tail call fastcc void @my_srand()
  %cmp7 = icmp eq i64 %size, 0
  br i1 %cmp7, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.08 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i64, i64* %arr, i64 %i.08
  store i64 %i.08, i64* %arrayidx, align 8, !tbaa !8
  %inc = add nuw i64 %i.08, 1
  %exitcond = icmp eq i64 %inc, %size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  tail call void @scramble_array(i64* %arr, i64 %size)
  ret void
}

; Function Attrs: inlinehint norecurse nounwind uwtable writeonly
define internal fastcc void @my_srand() unnamed_addr #6 {
entry:
  store i64 1, i64* @rand_nxt, align 8, !tbaa !8
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local i32 @usage() local_unnamed_addr #7 {
entry:
  %0 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !9
  %1 = tail call i64 @fwrite(i8* getelementptr inbounds ([69 x i8], [69 x i8]* @.str, i64 0, i64 0), i64 68, i64 1, %struct._IO_FILE* %0) #13
  %2 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !9
  %3 = tail call i64 @fwrite(i8* getelementptr inbounds ([69 x i8], [69 x i8]* @.str.1, i64 0, i64 0), i64 68, i64 1, %struct._IO_FILE* %2) #13
  %4 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !9
  %5 = tail call i64 @fwrite(i8* getelementptr inbounds ([70 x i8], [70 x i8]* @.str.2, i64 0, i64 0), i64 69, i64 1, %struct._IO_FILE* %4) #13
  %6 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !9
  %7 = tail call i64 @fwrite(i8* getelementptr inbounds ([70 x i8], [70 x i8]* @.str.3, i64 0, i64 0), i64 69, i64 1, %struct._IO_FILE* %6) #13
  %8 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !9
  %9 = tail call i64 @fwrite(i8* getelementptr inbounds ([71 x i8], [71 x i8]* @.str.4, i64 0, i64 0), i64 70, i64 1, %struct._IO_FILE* %8) #13
  %10 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !9
  %11 = tail call i64 @fwrite(i8* getelementptr inbounds ([38 x i8], [38 x i8]* @.str.5, i64 0, i64 0), i64 37, i64 1, %struct._IO_FILE* %10) #13
  ret i32 -1
}

; Function Attrs: nounwind
declare dso_local i32 @fprintf(%struct._IO_FILE* nocapture, i8* nocapture readonly, ...) local_unnamed_addr #8

; Function Attrs: nounwind uwtable
define dso_local i32 @main(i32 %argc, i8** %argv) local_unnamed_addr #7 {
entry:
  %size = alloca i64, align 8
  %benchmark = alloca i32, align 4
  %help = alloca i32, align 4
  %check = alloca i32, align 4
  %t1 = alloca %struct.timeval, align 8
  %t2 = alloca %struct.timeval, align 8
  %0 = bitcast i64* %size to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #12
  %1 = bitcast i32* %benchmark to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %1) #12
  %2 = bitcast i32* %help to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %2) #12
  %3 = bitcast i32* %check to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %3) #12
  store i32 0, i32* %check, align 4, !tbaa !11
  store i64 3000000, i64* %size, align 8, !tbaa !8
  call void (i32, i8**, i8**, i32*, ...) @get_options(i32 %argc, i8** %argv, i8** getelementptr inbounds ([5 x i8*], [5 x i8*]* @specifiers, i64 0, i64 0), i32* getelementptr inbounds ([5 x i32], [5 x i32]* @opt_types, i64 0, i64 0), i64* nonnull %size, i32* nonnull %check, i32* nonnull %benchmark, i32* nonnull %help) #12
  %4 = load i32, i32* %help, align 4, !tbaa !11
  %tobool = icmp eq i32 %4, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %call = call i32 @usage()
  br label %cleanup

if.end:                                           ; preds = %entry
  %5 = load i32, i32* %benchmark, align 4, !tbaa !11
  switch i32 %5, label %if.end5 [
    i32 3, label %sw.bb4
    i32 1, label %sw.bb
    i32 2, label %sw.bb3
  ]

sw.bb:                                            ; preds = %if.end
  store i64 10000, i64* %size, align 8, !tbaa !8
  br label %if.end5

sw.bb3:                                           ; preds = %if.end
  store i64 3000000, i64* %size, align 8, !tbaa !8
  br label %if.end5

sw.bb4:                                           ; preds = %if.end
  store i64 4100000, i64* %size, align 8, !tbaa !8
  br label %if.end5

if.end5:                                          ; preds = %if.end, %sw.bb, %sw.bb3, %sw.bb4
  %6 = load i64, i64* %size, align 8, !tbaa !8
  %mul = shl i64 %6, 3
  %call6 = call noalias i8* @malloc(i64 %mul) #12
  %7 = bitcast i8* %call6 to i64*
  %call8 = call noalias i8* @malloc(i64 %mul) #12
  %8 = bitcast i8* %call8 to i64*
  call void @fill_array(i64* %7, i64 %6)
  %9 = bitcast %struct.timeval* %t1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %9) #12
  %10 = bitcast %struct.timeval* %t2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %10) #12
  %call9 = call i32 @gettimeofday(%struct.timeval* nonnull %t1, %struct.timezone* null) #12
  %11 = load i64, i64* %size, align 8, !tbaa !8
  call void @cilksort(i64* %7, i64* %8, i64 %11)
  %call10 = call i32 @gettimeofday(%struct.timeval* nonnull %t2, %struct.timezone* null) #12
  %call11 = call i64 @todval(%struct.timeval* nonnull %t2)
  %call12 = call i64 @todval(%struct.timeval* nonnull %t1)
  %sub = sub i64 %call11, %call12
  %div = udiv i64 %sub, 1000
  %conv = uitofp i64 %div to double
  %div13 = fdiv double %conv, 1.000000e+03
  %call14 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.10, i64 0, i64 0), double %div13)
  %12 = load i32, i32* %check, align 4, !tbaa !11
  %tobool15 = icmp eq i32 %12, 0
  br i1 %tobool15, label %if.end28, label %if.then16

if.then16:                                        ; preds = %if.end5
  %puts = call i32 @puts(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @str, i64 0, i64 0))
  %13 = load i64, i64* %size, align 8, !tbaa !8
  %cmp51 = icmp sgt i64 %13, 0
  br i1 %cmp51, label %for.body.lr.ph, label %for.end.thread

for.end.thread:                                   ; preds = %if.then16
  %14 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !9
  br label %if.else

for.body.lr.ph:                                   ; preds = %if.then16
  %15 = load i64, i64* %size, align 8, !tbaa !8
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %success.053 = phi i32 [ 1, %for.body.lr.ph ], [ %spec.select, %for.body ]
  %i.052 = phi i64 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds i64, i64* %7, i64 %i.052
  %16 = load i64, i64* %arrayidx, align 8, !tbaa !8
  %cmp19 = icmp eq i64 %16, %i.052
  %spec.select = select i1 %cmp19, i32 %success.053, i32 0
  %inc = add nuw nsw i64 %i.052, 1
  %cmp = icmp slt i64 %inc, %15
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %phitmp = icmp eq i32 %spec.select, 0
  %17 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !9
  br i1 %phitmp, label %if.then24, label %if.else

if.then24:                                        ; preds = %for.end
  %18 = call i64 @fwrite(i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str.12, i64 0, i64 0), i64 16, i64 1, %struct._IO_FILE* %17) #13
  br label %if.end28

if.else:                                          ; preds = %for.end.thread, %for.end
  %19 = phi %struct._IO_FILE* [ %14, %for.end.thread ], [ %17, %for.end ]
  %20 = call i64 @fwrite(i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str.13, i64 0, i64 0), i64 19, i64 1, %struct._IO_FILE* %19) #13
  br label %if.end28

if.end28:                                         ; preds = %if.end5, %if.then24, %if.else
  %21 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !9
  %22 = call i64 @fwrite(i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.14, i64 0, i64 0), i64 24, i64 1, %struct._IO_FILE* %21) #13
  %23 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8, !tbaa !9
  %24 = load i64, i64* %size, align 8, !tbaa !8
  %call30 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %23, i8* getelementptr inbounds ([36 x i8], [36 x i8]* @.str.15, i64 0, i64 0), i64 %24) #13
  call void @free(i8* %call6) #12
  call void @free(i8* %call8) #12
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %10) #12
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %9) #12
  br label %cleanup

cleanup:                                          ; preds = %if.end28, %if.then
  %retval.0 = phi i32 [ -1, %if.then ], [ 0, %if.end28 ]
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %3) #12
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %2) #12
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %1) #12
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #12
  ret i32 %retval.0
}

declare dso_local void @get_options(i32, i8**, i8**, i32*, ...) local_unnamed_addr #9

; Function Attrs: nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #8

; Function Attrs: nounwind
declare dso_local i32 @gettimeofday(%struct.timeval* nocapture, %struct.timezone* nocapture) local_unnamed_addr #8

; Function Attrs: nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #8

; Function Attrs: nounwind
declare dso_local void @free(i8* nocapture) local_unnamed_addr #8

; Function Attrs: argmemonly inlinehint norecurse nounwind readonly uwtable
define internal fastcc i64 @choose_pivot(i64* %low, i64* %high) unnamed_addr #10 {
entry:
  %0 = load i64, i64* %low, align 8, !tbaa !8
  %1 = load i64, i64* %high, align 8, !tbaa !8
  %sub.ptr.lhs.cast = ptrtoint i64* %high to i64
  %sub.ptr.rhs.cast = ptrtoint i64* %low to i64
  %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  %sub.ptr.div = ashr exact i64 %sub.ptr.sub, 3
  %div = sdiv i64 %sub.ptr.div, 2
  %arrayidx = getelementptr inbounds i64, i64* %low, i64 %div
  %2 = load i64, i64* %arrayidx, align 8, !tbaa !8
  %call = tail call fastcc i64 @med3(i64 %0, i64 %1, i64 %2)
  ret i64 %call
}

; Function Attrs: inlinehint norecurse nounwind readnone uwtable
define internal fastcc i64 @med3(i64 %a, i64 %b, i64 %c) unnamed_addr #11 {
entry:
  %cmp = icmp slt i64 %a, %b
  br i1 %cmp, label %if.then, label %if.else6

if.then:                                          ; preds = %entry
  %cmp1 = icmp slt i64 %b, %c
  br i1 %cmp1, label %cleanup, label %if.else

if.else:                                          ; preds = %if.then
  %cmp3 = icmp slt i64 %a, %c
  %c.a = select i1 %cmp3, i64 %c, i64 %a
  br label %cleanup

if.else6:                                         ; preds = %entry
  %cmp7 = icmp sgt i64 %b, %c
  br i1 %cmp7, label %cleanup, label %if.else9

if.else9:                                         ; preds = %if.else6
  %cmp10 = icmp sgt i64 %a, %c
  %c.a24 = select i1 %cmp10, i64 %c, i64 %a
  br label %cleanup

cleanup:                                          ; preds = %if.else9, %if.else6, %if.else, %if.then
  %retval.0 = phi i64 [ %b, %if.then ], [ %c.a, %if.else ], [ %b, %if.else6 ], [ %c.a24, %if.else9 ]
  ret i64 %retval.0
}

; Function Attrs: nounwind
declare i64 @fwrite(i8* nocapture, i64, i64, %struct._IO_FILE* nocapture) local_unnamed_addr #12

; Function Attrs: nounwind
declare i32 @puts(i8* nocapture readonly) local_unnamed_addr #12

attributes #0 = { argmemonly norecurse nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { argmemonly nounwind }
attributes #3 = { argmemonly norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { inlinehint norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { inlinehint norecurse nounwind uwtable writeonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #10 = { argmemonly inlinehint norecurse nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #11 = { inlinehint norecurse nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #12 = { nounwind }
attributes #13 = { cold }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"wchar_size", i32 4}
!2 = !{!3, !4, i64 0}
!3 = !{!"timeval", !4, i64 0, !4, i64 8}
!4 = !{!"long", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!3, !4, i64 8}
!8 = !{!4, !4, i64 0}
!9 = !{!10, !10, i64 0}
!10 = !{!"any pointer", !5, i64 0}
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !5, i64 0}
