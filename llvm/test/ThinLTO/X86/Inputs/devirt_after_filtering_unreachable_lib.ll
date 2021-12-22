; ModuleID = 'lib.cc'
source_filename = "lib.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%class.Derived = type { %class.Base }
%class.Base = type { i32 (...)** }

$_ZN7DerivedD2Ev = comdat any

$_ZN7DerivedD0Ev = comdat any

$_ZN4BaseD2Ev = comdat any

$_ZN4BaseD0Ev = comdat any

$_ZTS4Base = comdat any

$_ZTI4Base = comdat any

$_ZTV4Base = comdat any

@_ZTV7Derived = hidden unnamed_addr constant { [5 x i8*] } { [5 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTI7Derived to i8*), i8* bitcast (void (%class.Derived*)* @_ZN7DerivedD2Ev to i8*), i8* bitcast (void (%class.Derived*)* @_ZN7DerivedD0Ev to i8*), i8* bitcast (void (%class.Derived*)* @_ZN7Derived1xEv to i8*)] }, align 8, !type !0, !type !1, !type !2, !type !3, !vcall_visibility !4
@_ZTVN10__cxxabiv120__si_class_type_infoE = external dso_local global i8*
@_ZTS7Derived = hidden constant [9 x i8] c"7Derived\00", align 1
@_ZTVN10__cxxabiv117__class_type_infoE = external dso_local global i8*
@_ZTS4Base = linkonce_odr hidden constant [6 x i8] c"4Base\00", comdat, align 1
@_ZTI4Base = linkonce_odr hidden constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @_ZTS4Base, i32 0, i32 0) }, comdat, align 8
@_ZTI7Derived = hidden constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([9 x i8], [9 x i8]* @_ZTS7Derived, i32 0, i32 0), i8* bitcast ({ i8*, i8* }* @_ZTI4Base to i8*) }, align 8
@_ZTV4Base = linkonce_odr hidden unnamed_addr constant { [5 x i8*] } { [5 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI4Base to i8*), i8* bitcast (void (%class.Base*)* @_ZN4BaseD2Ev to i8*), i8* bitcast (void (%class.Base*)* @_ZN4BaseD0Ev to i8*), i8* bitcast (void ()* @__cxa_pure_virtual to i8*)] }, comdat, align 8, !type !0, !type !1, !vcall_visibility !4
@str = private unnamed_addr constant [11 x i8] c"Derived::x\00", align 1
@str.3 = private unnamed_addr constant [23 x i8] c"In Derived::~Derived()\00", align 1
@str.4 = private unnamed_addr constant [17 x i8] c"In Base::~Base()\00", align 1

; Function Attrs: mustprogress nofree nounwind uwtable
define hidden void @_ZN7Derived1xEv(%class.Derived* nocapture nonnull readnone align 8 %this) unnamed_addr align 2 {
entry:
  %puts = tail call i32 @puts(i8* nonnull dereferenceable(1) getelementptr inbounds ([11 x i8], [11 x i8]* @str, i64 0, i64 0))
  ret void
}

; Function Attrs: mustprogress nounwind uwtable
define hidden void @_Z3fooP4Base(%class.Base* %b) local_unnamed_addr {
entry:
  %isnull = icmp eq %class.Base* %b, null
  br i1 %isnull, label %delete.end, label %delete.notnull

delete.notnull:                                   ; preds = %entry
  %0 = bitcast %class.Base* %b to void (%class.Base*)***
  %vtable = load void (%class.Base*)**, void (%class.Base*)*** %0, align 8, !tbaa !9
  %1 = bitcast void (%class.Base*)** %vtable to i8*
  %2 = tail call i1 @llvm.type.test(i8* %1, metadata !"_ZTS4Base")
  tail call void @llvm.assume(i1 %2)
  %vfn = getelementptr inbounds void (%class.Base*)*, void (%class.Base*)** %vtable, i64 1
  %3 = load void (%class.Base*)*, void (%class.Base*)** %vfn, align 8
  tail call void %3(%class.Base* nonnull align 8 dereferenceable(8) %b)
  br label %delete.end

delete.end:                                       ; preds = %delete.notnull, %entry
  ret void
}

declare i1 @llvm.type.test(i8*, metadata)

declare void @llvm.assume(i1 noundef)

define linkonce_odr hidden void @_ZN7DerivedD2Ev(%class.Derived* nonnull align 8 dereferenceable(8) %this) unnamed_addr comdat align 2 {
entry:
  %0 = getelementptr inbounds %class.Derived, %class.Derived* %this, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [5 x i8*] }, { [5 x i8*] }* @_ZTV7Derived, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %0, align 8, !tbaa !9
  %puts = tail call i32 @puts(i8* nonnull dereferenceable(1) getelementptr inbounds ([23 x i8], [23 x i8]* @str.3, i64 0, i64 0))
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [5 x i8*] }, { [5 x i8*] }* @_ZTV4Base, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %0, align 8, !tbaa !9
  %puts.i = tail call i32 @puts(i8* nonnull dereferenceable(1) getelementptr inbounds ([17 x i8], [17 x i8]* @str.4, i64 0, i64 0))
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr hidden void @_ZN7DerivedD0Ev(%class.Derived* nonnull align 8 dereferenceable(8) %this) unnamed_addr comdat align 2 {
entry:
  %0 = getelementptr inbounds %class.Derived, %class.Derived* %this, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [5 x i8*] }, { [5 x i8*] }* @_ZTV7Derived, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %0, align 8, !tbaa !9
  %puts.i = tail call i32 @puts(i8* nonnull dereferenceable(1) getelementptr inbounds ([23 x i8], [23 x i8]* @str.3, i64 0, i64 0))
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [5 x i8*] }, { [5 x i8*] }* @_ZTV4Base, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %0, align 8, !tbaa !9
  %puts.i.i = tail call i32 @puts(i8* nonnull dereferenceable(1) getelementptr inbounds ([17 x i8], [17 x i8]* @str.4, i64 0, i64 0))
  %1 = bitcast %class.Derived* %this to i8*
  tail call void @_ZdlPv(i8* nonnull %1)
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr hidden void @_ZN4BaseD2Ev(%class.Base* nonnull align 8 dereferenceable(8) %this) unnamed_addr comdat align 2 {
entry:
  %0 = getelementptr inbounds %class.Base, %class.Base* %this, i64 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [5 x i8*] }, { [5 x i8*] }* @_ZTV4Base, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %0, align 8, !tbaa !9
  %puts = tail call i32 @puts(i8* nonnull dereferenceable(1) getelementptr inbounds ([17 x i8], [17 x i8]* @str.4, i64 0, i64 0))
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr hidden void @_ZN4BaseD0Ev(%class.Base* nonnull align 8 dereferenceable(8) %this) unnamed_addr comdat align 2 {
entry:
  tail call void @llvm.trap()
  unreachable
}

declare dso_local void @__cxa_pure_virtual() unnamed_addr

declare void @llvm.trap()

declare dso_local void @_ZdlPv(i8*) local_unnamed_addr

declare noundef i32 @puts(i8* nocapture noundef readonly) local_unnamed_addr

!llvm.module.flags = !{!5, !6, !7}

!0 = !{i64 16, !"_ZTS4Base"}
!1 = !{i64 32, !"_ZTSM4BaseFvvE.virtual"}
!2 = !{i64 16, !"_ZTS7Derived"}
!3 = !{i64 32, !"_ZTSM7DerivedFvvE.virtual"}
!4 = !{i64 1}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 1, !"Virtual Function Elim", i32 0}
!7 = !{i32 7, !"uwtable", i32 1}
!9 = !{!10, !10, i64 0}
!10 = !{!"vtable pointer", !11, i64 0}
!11 = !{!"Simple C++ TBAA"}
