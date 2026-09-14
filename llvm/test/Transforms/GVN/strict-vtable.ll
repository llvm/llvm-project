; RUN: opt -passes=gvn -S %s -o /dev/null

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"

$_ZTV1i = comdat any

$_ZTI1i = comdat any

$_ZTS1i = comdat any

@f = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@_ZZ1gvE1a = internal global { ptr, ptr } { ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr], [3 x ptr] }, ptr @_ZTV1i, i32 0, i32 0, i32 2), ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr], [3 x ptr] }, ptr @_ZTV1i, i32 0, i32 1, i32 2) }, align 8
@_ZTV1i = linkonce_odr dso_local constant { [3 x ptr], [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI1i, ptr @_ZN1b1cEv], [3 x ptr] [ptr inttoptr (i64 -8 to ptr), ptr @_ZTI1i, ptr @_ZN1d1eEf] }, comdat, align 8
@_ZTI1i = linkonce_odr dso_local constant { ptr, ptr, i32, i32, ptr, i64, ptr, i64 } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv121__vmi_class_type_infoE, i64 2), ptr @_ZTS1i, i32 0, i32 2, ptr @_ZTI1b, i64 2, ptr @_ZTI1d, i64 2050 }, comdat, align 8
@_ZTVN10__cxxabiv121__vmi_class_type_infoE = external global [0 x ptr]
@_ZTS1i = linkonce_odr dso_local constant [3 x i8] c"1i\00", comdat, align 1
@_ZTI1b = external constant ptr
@_ZTI1d = external constant ptr

define dso_local void @_Z1hv() local_unnamed_addr {
entry:
  %0 = load float, ptr @f, align 4
  %vtable = load ptr, ptr getelementptr inbounds nuw (i8, ptr @_ZZ1gvE1a, i64 8), align 8, !invariant.group !0
  %1 = load ptr, ptr %vtable, align 8, !invariant.load !0
  tail call void %1(ptr noundef nonnull align 8 dereferenceable(8) getelementptr inbounds nuw (i8, ptr @_ZZ1gvE1a, i64 8), float noundef %0)
  ret void
}

define dso_local noundef nonnull align 8 dereferenceable(8) ptr @_Z1gv() local_unnamed_addr {
entry:
  ret ptr getelementptr inbounds nuw (i8, ptr @_ZZ1gvE1a, i64 8)
}

declare noundef i32 @_ZN1b1cEv(ptr noundef nonnull align 8 dereferenceable(8)) unnamed_addr

declare void @_ZN1d1eEf(ptr noundef nonnull align 8 dereferenceable(8), float noundef) unnamed_addr

!0 = !{}
