; RUN: llc < %s -mtriple=x86_64-w64-windows-gnu | FileCheck %s
; Control Flow Guard is currently only available on Windows

; This file was generated from the following source, using this command line:
; clang++ -target x86_64-w64-windows-gnu cfguard-mingw.cpp -S -emit-llvm -o cfguard-mingw.ll -O -Xclang -cfguard
;
;-------------------------------------------------------------------------------
; class __attribute__((dllexport)) Base {
; public:
;     __attribute__((dllexport)) Base() = default;
;     __attribute__((dllexport)) virtual ~Base() = default;
;     __attribute__((dllexport)) virtual int calc() const {
;         return m_field * 2;
;     }
;     int m_field{0};
; };
;
; class __attribute__((dllexport)) Derived : public Base {
; public:
;     __attribute__((dllexport)) Derived() = default;
;     __attribute__((dllexport)) ~Derived() override = default;
;     __attribute__((dllexport)) int calc() const override {
;         return m_field * 2 + m_newfield;
;     }
;     int m_newfield{0};
; };
;
; __attribute((noinline)) void address_taken() {}
; __attribute((noinline)) void address_not_taken() {}
;
; using fn_t = void (*)();
; __attribute__((dllexport)) fn_t get_address() {
;     address_not_taken();
;     return &address_taken;
; }
;-------------------------------------------------------------------------------

; CHECK: @feat.00 = 2048

; CHECK: .section .gfids$y
; CHECK: .symidx _ZNK7Derived4calcEv
; CHECK: .symidx _Z13address_takenv

; ModuleID = 'cfguard-mingw.cpp'
source_filename = "cfguard-mingw.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-w64-windows-gnu"

%class.Base = type <{ ptr, i32, [4 x i8] }>
%class.Derived = type { %class.Base.base, i32 }
%class.Base.base = type <{ ptr, i32 }>

$_ZN4BaseC2Ev = comdat any

$_ZN4BaseC1Ev = comdat any

$_ZNK4Base4calcEv = comdat any

$_ZN4BaseD2Ev = comdat any

$_ZN4BaseD1Ev = comdat any

$_ZN4BaseD0Ev = comdat any

$_ZN7DerivedC2Ev = comdat any

$_ZN7DerivedC1Ev = comdat any

$_ZNK7Derived4calcEv = comdat any

$_ZN7DerivedD2Ev = comdat any

$_ZN7DerivedD1Ev = comdat any

$_ZN7DerivedD0Ev = comdat any

$_ZTV4Base = comdat any

$_ZTV7Derived = comdat any

$_ZTS4Base = comdat any

$_ZTI4Base = comdat any

$_ZTS7Derived = comdat any

$_ZTI7Derived = comdat any

@_ZTV4Base = weak_odr dso_local dllexport unnamed_addr constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr @_ZTI4Base, ptr @_ZN4BaseD1Ev, ptr @_ZN4BaseD0Ev, ptr @_ZNK4Base4calcEv] }, comdat, align 8
@_ZTV7Derived = weak_odr dso_local dllexport unnamed_addr constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr @_ZTI7Derived, ptr @_ZN7DerivedD1Ev, ptr @_ZN7DerivedD0Ev, ptr @_ZNK7Derived4calcEv] }, comdat, align 8
@_ZTVN10__cxxabiv117__class_type_infoE = external global ptr
@_ZTS4Base = linkonce_odr dso_local constant [6 x i8] c"4Base\00", comdat, align 1
@_ZTI4Base = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS4Base }, comdat, align 8
@_ZTVN10__cxxabiv120__si_class_type_infoE = external global ptr
@_ZTS7Derived = linkonce_odr dso_local constant [9 x i8] c"7Derived\00", comdat, align 1
@_ZTI7Derived = linkonce_odr dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTS7Derived, ptr @_ZTI4Base }, comdat, align 8

; Function Attrs: nounwind uwtable
define weak_odr dso_local dllexport void @_ZN4BaseC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %0) unnamed_addr #0 comdat align 2 {
  store ptr getelementptr inbounds ({ [5 x ptr] }, ptr @_ZTV4Base, i64 0, i32 0, i64 2), ptr %0, align 8, !tbaa !5
  %2 = getelementptr inbounds %class.Base, ptr %0, i64 0, i32 1
  store i32 0, ptr %2, align 8, !tbaa !8
  ret void
}

; Function Attrs: nounwind uwtable
define weak_odr dso_local dllexport void @_ZN4BaseC1Ev(ptr noundef nonnull align 8 dereferenceable(12) %0) unnamed_addr #0 comdat align 2 {
  store ptr getelementptr inbounds ({ [5 x ptr] }, ptr @_ZTV4Base, i64 0, i32 0, i64 2), ptr %0, align 8, !tbaa !5
  %2 = getelementptr inbounds %class.Base, ptr %0, i64 0, i32 1
  store i32 0, ptr %2, align 8, !tbaa !8
  ret void
}

; Function Attrs: mustprogress nounwind uwtable
define weak_odr dso_local dllexport noundef i32 @_ZNK4Base4calcEv(ptr noundef nonnull align 8 dereferenceable(12) %0) unnamed_addr #1 comdat align 2 {
  %2 = getelementptr inbounds %class.Base, ptr %0, i64 0, i32 1
  %3 = load i32, ptr %2, align 8, !tbaa !8
  %4 = shl nsw i32 %3, 1
  ret i32 %4
}

; Function Attrs: nounwind uwtable
define weak_odr dso_local dllexport void @_ZN4BaseD2Ev(ptr noundef nonnull align 8 dereferenceable(12) %0) unnamed_addr #0 comdat align 2 {
  ret void
}

; Function Attrs: nounwind uwtable
define weak_odr dso_local dllexport void @_ZN4BaseD1Ev(ptr noundef nonnull align 8 dereferenceable(12) %0) unnamed_addr #0 comdat align 2 {
  ret void
}

; Function Attrs: nounwind uwtable
define weak_odr dso_local dllexport void @_ZN4BaseD0Ev(ptr noundef nonnull align 8 dereferenceable(12) %0) unnamed_addr #0 comdat align 2 {
  tail call void @_ZdlPv(ptr noundef nonnull %0) #5
  ret void
}

; Function Attrs: nobuiltin nounwind
declare dso_local void @_ZdlPv(ptr noundef) local_unnamed_addr #2

; Function Attrs: nounwind uwtable
define weak_odr dso_local dllexport void @_ZN7DerivedC2Ev(ptr noundef nonnull align 8 dereferenceable(16) %0) unnamed_addr #0 comdat align 2 {
  store ptr getelementptr inbounds ({ [5 x ptr] }, ptr @_ZTV4Base, i64 0, i32 0, i64 2), ptr %0, align 8, !tbaa !5
  %2 = getelementptr inbounds %class.Base, ptr %0, i64 0, i32 1
  store i32 0, ptr %2, align 8, !tbaa !8
  store ptr getelementptr inbounds ({ [5 x ptr] }, ptr @_ZTV7Derived, i64 0, i32 0, i64 2), ptr %0, align 8, !tbaa !5
  %3 = getelementptr inbounds %class.Derived, ptr %0, i64 0, i32 1
  store i32 0, ptr %3, align 4, !tbaa !12
  ret void
}

; Function Attrs: nounwind uwtable
define weak_odr dso_local dllexport void @_ZN7DerivedC1Ev(ptr noundef nonnull align 8 dereferenceable(16) %0) unnamed_addr #0 comdat align 2 {
  store ptr getelementptr inbounds ({ [5 x ptr] }, ptr @_ZTV4Base, i64 0, i32 0, i64 2), ptr %0, align 8, !tbaa !5
  %2 = getelementptr inbounds %class.Base, ptr %0, i64 0, i32 1
  store i32 0, ptr %2, align 8, !tbaa !8
  store ptr getelementptr inbounds ({ [5 x ptr] }, ptr @_ZTV7Derived, i64 0, i32 0, i64 2), ptr %0, align 8, !tbaa !5
  %3 = getelementptr inbounds %class.Derived, ptr %0, i64 0, i32 1
  store i32 0, ptr %3, align 4, !tbaa !12
  ret void
}

; Function Attrs: mustprogress nounwind uwtable
define weak_odr dso_local dllexport noundef i32 @_ZNK7Derived4calcEv(ptr noundef nonnull align 8 dereferenceable(16) %0) unnamed_addr #1 comdat align 2 {
  %2 = getelementptr inbounds %class.Base, ptr %0, i64 0, i32 1
  %3 = load i32, ptr %2, align 8, !tbaa !8
  %4 = shl nsw i32 %3, 1
  %5 = getelementptr inbounds %class.Derived, ptr %0, i64 0, i32 1
  %6 = load i32, ptr %5, align 4, !tbaa !12
  %7 = add nsw i32 %4, %6
  ret i32 %7
}

; Function Attrs: nounwind uwtable
define weak_odr dso_local dllexport void @_ZN7DerivedD2Ev(ptr noundef nonnull align 8 dereferenceable(16) %0) unnamed_addr #0 comdat align 2 {
  ret void
}

; Function Attrs: nounwind uwtable
define weak_odr dso_local dllexport void @_ZN7DerivedD1Ev(ptr noundef nonnull align 8 dereferenceable(16) %0) unnamed_addr #0 comdat align 2 {
  ret void
}

; Function Attrs: nounwind uwtable
define weak_odr dso_local dllexport void @_ZN7DerivedD0Ev(ptr noundef nonnull align 8 dereferenceable(16) %0) unnamed_addr #0 comdat align 2 {
  tail call void @_ZdlPv(ptr noundef nonnull %0) #5
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind readnone willreturn uwtable
define dso_local void @_Z13address_takenv() #3 {
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind readnone willreturn uwtable
define dso_local void @_Z17address_not_takenv() local_unnamed_addr #3 {
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn uwtable
define dso_local dllexport noundef nonnull ptr @_Z11get_addressv() local_unnamed_addr #4 {
  ret ptr @_Z13address_takenv
}

attributes #0 = { nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nobuiltin nounwind "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { mustprogress nofree noinline norecurse nosync nounwind readnone willreturn uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { mustprogress nofree norecurse nosync nounwind readnone willreturn uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { builtin nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 2, !"cfguard", i32 2}
!1 = !{i32 1, !"wchar_size", i32 2}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"clang version 16.0.0"}
!5 = !{!6, !6, i64 0}
!6 = !{!"vtable pointer", !7, i64 0}
!7 = !{!"Simple C++ TBAA"}
!8 = !{!9, !10, i64 8}
!9 = !{!"_ZTS4Base", !10, i64 8}
!10 = !{!"int", !11, i64 0}
!11 = !{!"omnipotent char", !7, i64 0}
!12 = !{!13, !10, i64 12}
!13 = !{!"_ZTS7Derived", !9, i64 0, !10, i64 12}
