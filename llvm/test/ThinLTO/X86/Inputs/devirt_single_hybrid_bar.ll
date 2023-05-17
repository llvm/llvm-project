; ModuleID = 'bar.cpp'
source_filename = "bar.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.A = type { ptr }

$_ZNK1A1fEv = comdat any

$_ZTV1A = comdat any

$_ZTS1A = comdat any

$_ZTI1A = comdat any

@_ZTV1A = linkonce_odr hidden unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI1A, ptr @_ZNK1A1fEv] }, comdat, align 8, !type !0, !type !1
@_ZTVN10__cxxabiv117__class_type_infoE = external dso_local global ptr
@_ZTS1A = linkonce_odr hidden constant [3 x i8] c"1A\00", comdat, align 1
@_ZTI1A = linkonce_odr hidden constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS1A }, comdat, align 8

; Function Attrs: uwtable
define hidden i32 @_Z3barv() local_unnamed_addr #0 {
entry:
  %b = alloca %struct.A, align 8
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %b)
  store ptr getelementptr inbounds ({ [3 x ptr] }, ptr @_ZTV1A, i64 0, inrange i32 0, i64 2), ptr %b, align 8, !tbaa !4
  %call = call i32 @_Z3fooP1A(ptr nonnull %b)
  %add = add nsw i32 %call, 10
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %b) #4
  ret i32 %add
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)

declare dso_local i32 @_Z3fooP1A(ptr) local_unnamed_addr

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)

; Function Attrs: nounwind uwtable
define linkonce_odr hidden i32 @_ZNK1A1fEv(ptr %this) unnamed_addr comdat align 2 {
entry:
  ret i32 3
}

!llvm.module.flags = !{!2}
!llvm.ident = !{!3}

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTSM1AKFivE.virtual"}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{!"clang version 10.0.0 (trunk 373596)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"vtable pointer", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
