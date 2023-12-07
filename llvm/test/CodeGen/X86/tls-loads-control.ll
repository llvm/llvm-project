; RUN: llc -mtriple=x86_64-unknown-unknown -O2 --relocation-model=pic --tls-load-hoist=true --stop-after=tlshoist -o - %s | FileCheck %s
; RUN: llc -mtriple=x86_64-unknown-unknown -O2 --relocation-model=pic --stop-after=tlshoist -o - %s | FileCheck %s

; This test come from compiling clang/test/CodeGen/intel/tls_loads.cpp with:
; (clang tls_loads.cpp -fPIC -ftls-model=global-dynamic -O2 -S -emit-llvm)

; // Variable declaration and definition:
; thread_local int thl_x;
; thread_local int thl_x2;
;
; struct SS {
;   char thl_c;
;   int num;
; };
;
; int gfunc();
; int gfunc2(int);

; // First function (@_Z2f1i):
; int f1(int c) {
;   while (c)
;     c++;
;
;   int *px = &thl_x;
;   c -= gfunc();
;
;   while(c++) {
;     c = gfunc();
;     while (c--)
;       *px += gfunc2(thl_x2);
;   }
;   return *px;
; }

$_ZTW5thl_x = comdat any

$_ZTW6thl_x2 = comdat any

@thl_x = thread_local global i32 0, align 4
@thl_x2 = thread_local global i32 0, align 4
@_ZZ2f2iE2st.0 = internal thread_local unnamed_addr global i8 0, align 4
@_ZZ2f2iE2st.1 = internal thread_local unnamed_addr global i32 0, align 4

; Function Attrs: mustprogress uwtable
define noundef i32 @_Z2f1i(i32 noundef %c) local_unnamed_addr #0 {
; CHECK-LABEL: _Z2f1i
; CHECK:      entry:
; CHECK-NEXT:   %call = tail call noundef i32 @_Z5gfuncv()
; CHECK-NEXT:   %phi.cmp = icmp eq i32 %call, 0
; CHECK-NEXT:   %tls_bitcast1 = bitcast ptr @thl_x to ptr
; CHECK-NEXT:   br i1 %phi.cmp, label %while.end11, label %while.body4.preheader

; CHECK:      while.body4.preheader:
; CHECK-NEXT:   %tls_bitcast = bitcast ptr @thl_x2 to ptr
; CHECK-NEXT:   br label %while.body4

; CHECK:      while.body4:
; CHECK-NEXT:   %call5 = tail call noundef i32 @_Z5gfuncv()
; CHECK-NEXT:   %tobool7.not18 = icmp eq i32 %call5, 0
; CHECK-NEXT:   br i1 %tobool7.not18, label %while.body4.backedge, label %while.body8.preheader

; CHECK:      while.body8.preheader:
; CHECK-NEXT:   br label %while.body8

; CHECK:      while.body4.backedge.loopexit:
; CHECK-NEXT:   br label %while.body4.backedge

; CHECK:      while.body4.backedge:
; CHECK-NEXT:   br label %while.body4, !llvm.loop !4

; CHECK:      while.body8:
; CHECK-NEXT:   %c.addr.219 = phi i32 [ %dec, %while.body8 ], [ %call5, %while.body8.preheader ]
; CHECK-NEXT:   %dec = add i32 %c.addr.219, -1
; CHECK-NEXT:   %0 = load i32, ptr %tls_bitcast, align 4
; CHECK-NEXT:   %call9 = tail call noundef i32 @_Z6gfunc2i(i32 noundef %0)
; CHECK-NEXT:   %1 = load i32, ptr %tls_bitcast1, align 4
; CHECK-NEXT:   %add = add nsw i32 %1, %call9
; CHECK-NEXT:   store i32 %add, ptr %tls_bitcast1, align 4
; CHECK-NEXT:   %tobool7.not = icmp eq i32 %dec, 0
; CHECK-NEXT:   br i1 %tobool7.not, label %while.body4.backedge.loopexit, label %while.body8, !llvm.loop !4

; CHECK:      while.end11:
; CHECK-NEXT:   %2 = load i32, ptr %tls_bitcast1, align 4
; CHECK-NEXT:   ret i32 %2

entry:
  %call = tail call noundef i32 @_Z5gfuncv()
  %phi.cmp = icmp eq i32 %call, 0
  br i1 %phi.cmp, label %while.end11, label %while.body4

while.body4:                                      ; preds = %entry, %while.body4.backedge
  %call5 = tail call noundef i32 @_Z5gfuncv()
  %tobool7.not18 = icmp eq i32 %call5, 0
  br i1 %tobool7.not18, label %while.body4.backedge, label %while.body8

while.body4.backedge:                             ; preds = %while.body8, %while.body4
  br label %while.body4, !llvm.loop !4

while.body8:                                      ; preds = %while.body4, %while.body8
  %c.addr.219 = phi i32 [ %dec, %while.body8 ], [ %call5, %while.body4 ]
  %dec = add nsw i32 %c.addr.219, -1
  %0 = load i32, ptr @thl_x2, align 4
  %call9 = tail call noundef i32 @_Z6gfunc2i(i32 noundef %0)
  %1 = load i32, ptr @thl_x, align 4
  %add = add nsw i32 %1, %call9
  store i32 %add, ptr @thl_x, align 4
  %tobool7.not = icmp eq i32 %dec, 0
  br i1 %tobool7.not, label %while.body4.backedge, label %while.body8, !llvm.loop !4

while.end11:                                      ; preds = %entry
  %2 = load i32, ptr @thl_x, align 4
  ret i32 %2
}

; // Sencond function (@_Z2f2i):
; int f2(int c) {
;   thread_local struct SS st;
;   c += gfunc();
;   while (c--) {
;     thl_x += gfunc();
;     st.thl_c += (char)gfunc();
;     st.num += gfunc();
;   }
;   return thl_x;
; }
declare noundef i32 @_Z5gfuncv() local_unnamed_addr #1

declare noundef i32 @_Z6gfunc2i(i32 noundef) local_unnamed_addr #1

; Function Attrs: mustprogress uwtable
define noundef i32 @_Z2f2i(i32 noundef %c) local_unnamed_addr #0 {
; CHECK-LABEL: _Z2f2i
; CHECK:      entry:
; CHECK-NEXT:   %call = tail call noundef i32 @_Z5gfuncv()
; CHECK-NEXT:   %add = add nsw i32 %call, %c
; CHECK-NEXT:   %tobool.not12 = icmp eq i32 %add, 0
; CHECK-NEXT:   %tls_bitcast = bitcast ptr @thl_x to ptr
; CHECK-NEXT:   br i1 %tobool.not12, label %while.end, label %while.body.preheader

; CHECK:      while.body.preheader:
; CHECK-NEXT:   %tls_bitcast1 = bitcast ptr @_ZZ2f2iE2st.0 to ptr
; CHECK-NEXT:   %tls_bitcast2 = bitcast ptr @_ZZ2f2iE2st.1 to ptr
; CHECK-NEXT:   br label %while.body

; CHECK:      while.body:
; CHECK-NEXT:   %c.addr.013 = phi i32 [ %dec, %while.body ], [ %add, %while.body.preheader ]
; CHECK-NEXT:   %dec = add i32 %c.addr.013, -1
; CHECK-NEXT:   %call1 = tail call noundef i32 @_Z5gfuncv()
; CHECK-NEXT:   %0 = load i32, ptr %tls_bitcast, align 4
; CHECK-NEXT:   %add2 = add nsw i32 %0, %call1
; CHECK-NEXT:   store i32 %add2, ptr %tls_bitcast, align 4
; CHECK-NEXT:   %call3 = tail call noundef i32 @_Z5gfuncv()
; CHECK-NEXT:   %1 = load i8, ptr %tls_bitcast1, align 4
; CHECK-NEXT:   %2 = trunc i32 %call3 to i8
; CHECK-NEXT:   %conv7 = add i8 %1, %2
; CHECK-NEXT:   store i8 %conv7, ptr %tls_bitcast1, align 4
; CHECK-NEXT:   %call8 = tail call noundef i32 @_Z5gfuncv()
; CHECK-NEXT:   %3 = load i32, ptr %tls_bitcast2, align 4
; CHECK-NEXT:   %add9 = add nsw i32 %3, %call8
; CHECK-NEXT:   store i32 %add9, ptr %tls_bitcast2, align 4
; CHECK-NEXT:   %tobool.not = icmp eq i32 %dec, 0
; CHECK-NEXT:   br i1 %tobool.not, label %while.end.loopexit, label %while.body

; CHECK:      while.end.loopexit:
; CHECK-NEXT:   br label %while.end

; CHECK:      while.end:
; CHECK-NEXT:   %4 = load i32, ptr %tls_bitcast, align 4
; CHECK-NEXT:   ret i32 %4
entry:
  %call = tail call noundef i32 @_Z5gfuncv()
  %add = add nsw i32 %call, %c
  %tobool.not12 = icmp eq i32 %add, 0
  br i1 %tobool.not12, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %c.addr.013 = phi i32 [ %dec, %while.body ], [ %add, %entry ]
  %dec = add nsw i32 %c.addr.013, -1
  %call1 = tail call noundef i32 @_Z5gfuncv()
  %0 = load i32, ptr @thl_x, align 4
  %add2 = add nsw i32 %0, %call1
  store i32 %add2, ptr @thl_x, align 4
  %call3 = tail call noundef i32 @_Z5gfuncv()
  %1 = load i8, ptr @_ZZ2f2iE2st.0, align 4
  %2 = trunc i32 %call3 to i8
  %conv7 = add i8 %1, %2
  store i8 %conv7, ptr @_ZZ2f2iE2st.0, align 4
  %call8 = tail call noundef i32 @_Z5gfuncv()
  %3 = load i32, ptr @_ZZ2f2iE2st.1, align 4
  %add9 = add nsw i32 %3, %call8
  store i32 %add9, ptr @_ZZ2f2iE2st.1, align 4
  %tobool.not = icmp eq i32 %dec, 0
  br i1 %tobool.not, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  %4 = load i32, ptr @thl_x, align 4
  ret i32 %4
}

; // Third function (@_Z2f3i):
; int f3(int c) {
;   int *px = &thl_x;
;   gfunc2(*px);
;   gfunc2(*px);
;   return 1;
; }

; Function Attrs: mustprogress uwtable
define noundef i32 @_Z2f3i(i32 noundef %c) local_unnamed_addr #0 {
; CHECK-LABEL: _Z2f3i
; CHECK:      entry:
; CHECK-NEXT:   %tls_bitcast = bitcast ptr @thl_x to ptr
; CHECK-NEXT:   %0 = load i32, ptr %tls_bitcast, align 4
; CHECK-NEXT:   %call = tail call noundef i32 @_Z6gfunc2i(i32 noundef %0)
; CHECK-NEXT:   %1 = load i32, ptr %tls_bitcast, align 4
; CHECK-NEXT:   %call1 = tail call noundef i32 @_Z6gfunc2i(i32 noundef %1)
; CHECK-NEXT:   ret i32 1
entry:
  %0 = load i32, ptr @thl_x, align 4
  %call = tail call noundef i32 @_Z6gfunc2i(i32 noundef %0)
  %1 = load i32, ptr @thl_x, align 4
  %call1 = tail call noundef i32 @_Z6gfunc2i(i32 noundef %1)
  ret i32 1
}

; Function Attrs: uwtable
define weak_odr hidden noundef ptr @_ZTW5thl_x() local_unnamed_addr #2 comdat {
  ret ptr @thl_x
}

; Function Attrs: uwtable
define weak_odr hidden noundef ptr @_ZTW6thl_x2() local_unnamed_addr #2 comdat {
  ret ptr @thl_x2
}

attributes #0 = { mustprogress uwtable "tls-load-hoist" "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { uwtable "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{i32 7, !"uwtable", i32 2}
!3 = !{!"clang version 15.0.0"}
!4 = distinct !{!4, !5}
!5 = !{!"llvm.loop.mustprogress"}
