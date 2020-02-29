; RUN: opt -debugify -memcpyopt -check-debugify -S < %s 2>&1 | FileCheck %s

; CHECK: CheckModuleDebugify: PASS

; CHECK-LABEL: define {{.*}} @_Z3bar3Foo
; CHECK: [[target:%.*]] = load i8*, i8** bitcast (%struct.Foo** @a to i8**), align 8, !dbg
; CHECK: %tmpcast = bitcast i8* [[target]] to %struct.Foo*, !dbg

%struct.Foo = type { i64, i64, i64 }

@a = dso_local global %struct.Foo* null, align 8

define dso_local void @_Z3bar3Foo(%struct.Foo* byval(%struct.Foo) align 8 %0) {
entry:
  %agg.tmp = alloca %struct.Foo, align 8
  %1 = load i8*, i8** bitcast (%struct.Foo** @a to i8**), align 8
  %2 = bitcast %struct.Foo* %agg.tmp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(24) %2, i8* nonnull align 8 dereferenceable(24) %1, i64 24, i1 false)
  call void @_Z3bar3Foo(%struct.Foo* nonnull byval(%struct.Foo) align 8 %agg.tmp)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #0
