; RUN: opt < %s -passes=pgo-instr-gen -S | FileCheck %s --check-prefix=GEN
; RUN: opt < %s -passes=pgo-instr-gen,instrprof -S | FileCheck %s --check-prefix=LOWER

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$_ZTV7Derived = comdat any

@_ZTV7Derived = constant { [3 x ptr], [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI7Derived, ptr @_ZN5Base15func1Eii], [3 x ptr] [ptr inttoptr (i64 -8 to ptr), ptr @_ZTI7Derived, ptr @_ZN5Base25func2Eii] }, comdat, align 8, !type !0, !type !1, !type !2, !type !3, !type !4, !type !5, !type !6, !type !7, !type !8
@_ZTVN10__cxxabiv121__vmi_class_type_infoE = external global [0 x ptr]
@_ZTS7Derived = constant [9 x i8] c"7Derived\00", align 1
@_ZTI5Base1 = external constant ptr
@_ZTI5Base2 = external constant ptr
@_ZTI7Derived =  constant { ptr, ptr, i32, i32, ptr, i64, ptr, i64 } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv121__vmi_class_type_infoE, i64 2), ptr @_ZTS7Derived, i32 0, i32 2, ptr @_ZTI5Base1, i64 2, ptr @_ZTI5Base2, i64 2050 }, align 8
@_ZTV5Base1 = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI5Base1, ptr @_ZN5Base15func1Eii] }, align 8, !type !0, !type !1
@_ZTV5Base2 = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI5Base2, ptr @_ZN5Base25func2Eii] }, align 8, !type !9, !type !4
@llvm.compiler.used = appending global [2 x ptr] [ptr @_ZTV5Base1, ptr @_ZTV5Base2], section "llvm.metadata"

declare ptr @_Z10createTypei(i32)
declare i32 @_ZN5Base15func1Eii(ptr, i32, i32)
declare i32 @_ZN5Base25func2Eii(ptr, i32, i32)

; GEN: @__llvm_profile_raw_version = hidden constant i64 72057594037927945, comdat
; GEN: @__profn_test_vtable_value_profiling = private constant [27 x i8] c"test_vtable_value_profiling"

; LOWER: $__profvt__ZTV7Derived = comdat any
; LOWER: $__profvt__ZTV5Base1 = comdat nodeduplicate
; LOWER: $__profvt__ZTV5Base2 = comdat nodeduplicate
; LOWER: @__llvm_profile_raw_version = hidden constant i64 72057594037927945, comdat
; LOWER: @__profc_test_vtable_value_profiling = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", comdat, align 8
; LOWER: @__profvp_test_vtable_value_profiling = private global [4 x i64] zeroinitializer, section "__llvm_prf_vals", comdat($__profc_test_vtable_value_profiling), align 8
; LOWER: @__profd_test_vtable_value_profiling = private global { i64, i64, i64, ptr, ptr, i32, [3 x i16] } { i64 1593873508557585901, i64 567090795815895039, i64 sub (i64 ptrtoint (ptr @__profc_test_vtable_value_profiling to i64), i64 ptrtoint (ptr @__profd_test_vtable_value_profiling to i64)), ptr @test_vtable_value_profiling.local, ptr @__profvp_test_vtable_value_profiling, i32 1, [3 x i16] [i16 2, i16 0, i16 2] }, section "__llvm_prf_data", comdat($__profc_test_vtable_value_profiling), align 8
; LOWER: @__profvt__ZTV7Derived = global { i64, ptr, i32 } { i64 -4576307468236080025, ptr @_ZTV7Derived, i32 48 }, section "__llvm_prf_vtab", comdat, align 8
; LOWER: @__profvt__ZTV5Base1 = global { i64, ptr, i32 } { i64 3215870116411581797, ptr @_ZTV5Base1, i32 24 }, section "__llvm_prf_vtab", comdat, align 8
; LOWER: @__profvt__ZTV5Base2 = global { i64, ptr, i32 } { i64 8378219803387680050, ptr @_ZTV5Base2, i32 24 }, section "__llvm_prf_vtab", comdat, align 8
; LOWER: @__llvm_prf_vnodes = private global [10 x { i64, i64, ptr }] zeroinitializer, section "__llvm_prf_vnds", align 8
; LOWER: @__llvm_prf_nm = private constant [37 x i8] c"\1B#x\DA+I-.\89/+IL\CAI\8D/K\CC)M\8D/(\CAO\CB\CC\C9\CCK\07\00\9Ea\0BC", section "__llvm_prf_names", align 1
; LOWER: @__llvm_prf_vnm = private constant [34 x i8] c"\22 x\DA\8B\8F\0A\093wI-\CA,KMa\8C\07rL\9D\12\8BS\0D\11L#\00\C3\A2\0A\E9", section "__llvm_prf_vnames", align 1
; LOWER: @llvm.used = appending global [6 x ptr] [ptr @__profvt__ZTV7Derived, ptr @__profvt__ZTV5Base1, ptr @__profvt__ZTV5Base2, ptr @__llvm_prf_vnodes, ptr @__llvm_prf_nm, ptr @__llvm_prf_vnm], section "llvm.metadata"

define i32 @test_vtable_value_profiling(i32 %a, i32 %b, i32 %c) {
; GEN-LABEL: define i32 @test_vtable_value_profiling(
; GEN-SAME: i32 [[A:%.*]], i32 [[B:%.*]], i32 [[C:%.*]]) {
; GEN-NEXT:  entry:
; GEN-NEXT:    call void @llvm.instrprof.increment(ptr @__profn_test_vtable_value_profiling, i64 567090795815895039, i32 1, i32 0)
; GEN-NEXT:    [[CALL:%.*]] = tail call ptr @_Z10createTypei(i32 [[C]])
; GEN-NEXT:    [[ADD_PTR:%.*]] = getelementptr inbounds i8, ptr [[CALL]], i64 8
; GEN-NEXT:    [[VTABLE:%.*]] = load ptr, ptr [[ADD_PTR]], align 8
; GEN-NEXT:    [[TMP0:%.*]] = ptrtoint ptr [[VTABLE]] to i64
; GEN-NEXT:    call void @llvm.instrprof.value.profile(ptr @__profn_test_vtable_value_profiling, i64 567090795815895039, i64 [[TMP0]], i32 2, i32 0)
; GEN-NEXT:    [[VFUNC:%.*]] = load ptr, ptr [[VTABLE]], align 8
; GEN-NEXT:    [[TMP1:%.*]] = ptrtoint ptr [[VFUNC]] to i64
; GEN-NEXT:    call void @llvm.instrprof.value.profile(ptr @__profn_test_vtable_value_profiling, i64 567090795815895039, i64 [[TMP1]], i32 0, i32 0)
; GEN-NEXT:    [[CALL1:%.*]] = tail call i32 [[VFUNC]](ptr [[ADD_PTR]], i32 [[A]], i32 [[B]])
; GEN-NEXT:    [[VTABLE2:%.*]] = load ptr, ptr [[CALL]], align 8
; GEN-NEXT:    [[TMP2:%.*]] = ptrtoint ptr [[VTABLE2]] to i64
; GEN-NEXT:    call void @llvm.instrprof.value.profile(ptr @__profn_test_vtable_value_profiling, i64 567090795815895039, i64 [[TMP2]], i32 2, i32 1)
; GEN-NEXT:    [[VFUNC2:%.*]] = load ptr, ptr [[VTABLE2]], align 8
; GEN-NEXT:    [[TMP3:%.*]] = ptrtoint ptr [[VFUNC2]] to i64
; GEN-NEXT:    call void @llvm.instrprof.value.profile(ptr @__profn_test_vtable_value_profiling, i64 567090795815895039, i64 [[TMP3]], i32 0, i32 1)
; GEN-NEXT:    [[CALL4:%.*]] = tail call i32 [[VFUNC2]](ptr [[CALL]], i32 [[B]], i32 [[A]])
; GEN-NEXT:    [[ADD:%.*]] = add nsw i32 [[CALL4]], [[CALL1]]
; GEN-NEXT:    ret i32 [[ADD]]
;
; LOWER-LABEL: define i32 @test_vtable_value_profiling(
; LOWER-SAME: i32 [[A:%.*]], i32 [[B:%.*]], i32 [[C:%.*]]) {
; LOWER-NEXT:  entry:
; LOWER-NEXT:    [[PGOCOUNT:%.*]] = load i64, ptr @__profc_test_vtable_value_profiling, align 8
; LOWER-NEXT:    [[TMP0:%.*]] = add i64 [[PGOCOUNT]], 1
; LOWER-NEXT:    store i64 [[TMP0]], ptr @__profc_test_vtable_value_profiling, align 8
; LOWER-NEXT:    [[CALL:%.*]] = tail call ptr @_Z10createTypei(i32 [[C]])
; LOWER-NEXT:    [[ADD_PTR:%.*]] = getelementptr inbounds i8, ptr [[CALL]], i64 8
; LOWER-NEXT:    [[VTABLE:%.*]] = load ptr, ptr [[ADD_PTR]], align 8
; LOWER-NEXT:    [[TMP1:%.*]] = ptrtoint ptr [[VTABLE]] to i64
; LOWER-NEXT:    call void @__llvm_profile_instrument_target(i64 [[TMP1]], ptr @__profd_test_vtable_value_profiling, i32 2)
; LOWER-NEXT:    [[VFUNC:%.*]] = load ptr, ptr [[VTABLE]], align 8
; LOWER-NEXT:    [[TMP2:%.*]] = ptrtoint ptr [[VFUNC]] to i64
; LOWER-NEXT:    call void @__llvm_profile_instrument_target(i64 [[TMP2]], ptr @__profd_test_vtable_value_profiling, i32 0)
; LOWER-NEXT:    [[CALL1:%.*]] = tail call i32 [[VFUNC]](ptr [[ADD_PTR]], i32 [[A]], i32 [[B]])
; LOWER-NEXT:    [[VTABLE2:%.*]] = load ptr, ptr [[CALL]], align 8
; LOWER-NEXT:    [[TMP3:%.*]] = ptrtoint ptr [[VTABLE2]] to i64
; LOWER-NEXT:    call void @__llvm_profile_instrument_target(i64 [[TMP3]], ptr @__profd_test_vtable_value_profiling, i32 3)
; LOWER-NEXT:    [[VFUNC2:%.*]] = load ptr, ptr [[VTABLE2]], align 8
; LOWER-NEXT:    [[TMP4:%.*]] = ptrtoint ptr [[VFUNC2]] to i64
; LOWER-NEXT:    call void @__llvm_profile_instrument_target(i64 [[TMP4]], ptr @__profd_test_vtable_value_profiling, i32 1)
; LOWER-NEXT:    [[CALL4:%.*]] = tail call i32 [[VFUNC2]](ptr [[CALL]], i32 [[B]], i32 [[A]])
; LOWER-NEXT:    [[ADD:%.*]] = add nsw i32 [[CALL4]], [[CALL1]]
; LOWER-NEXT:    ret i32 [[ADD]]
;
entry:
  %call = tail call ptr @_Z10createTypei(i32 %c)
  %add.ptr = getelementptr inbounds i8, ptr %call, i64 8
  %vtable = load ptr, ptr %add.ptr, align 8
  %vfunc = load ptr, ptr %vtable, align 8
  %call1 = tail call i32 %vfunc(ptr %add.ptr, i32 %a, i32 %b)
  %vtable2 = load ptr, ptr %call, align 8
  %vfunc2 = load ptr, ptr %vtable2, align 8
  %call4 = tail call i32 %vfunc2(ptr %call, i32 %b, i32 %a)
  %add = add nsw i32 %call4, %call1
  ret i32 %add
}

!0 = !{i64 16, !"_ZTS5Base1"}
!1 = !{i64 16, !"_ZTSM5Base1FiiiE.virtual"}
!2 = !{i64 40, !"_ZTSM5Base1FiiiE.virtual"}
!3 = !{i64 40, !"_ZTS5Base2"}
!4 = !{i64 16, !"_ZTSM5Base2FiiiE.virtual"}
!5 = !{i64 40, !"_ZTSM5Base2FiiiE.virtual"}
!6 = !{i64 16, !"_ZTS7Derived"}
!7 = !{i64 16, !"_ZTSM7DerivedFiiiE.virtual"}
!8 = !{i64 40, !"_ZTSM7DerivedFiiiE.virtual"}
!9 = !{i64 16, !"_ZTS5Base2"}
;.
; GEN: attributes #[[ATTR0:[0-9]+]] = { nounwind }
;.
; LOWER: attributes #[[ATTR0:[0-9]+]] = { nounwind }
;.
; GEN: [[META0:![0-9]+]] = !{i64 16, !"_ZTS5Base1"}
; GEN: [[META1:![0-9]+]] = !{i64 16, !"_ZTSM5Base1FiiiE.virtual"}
; GEN: [[META2:![0-9]+]] = !{i64 40, !"_ZTSM5Base1FiiiE.virtual"}
; GEN: [[META3:![0-9]+]] = !{i64 40, !"_ZTS5Base2"}
; GEN: [[META4:![0-9]+]] = !{i64 16, !"_ZTSM5Base2FiiiE.virtual"}
; GEN: [[META5:![0-9]+]] = !{i64 40, !"_ZTSM5Base2FiiiE.virtual"}
; GEN: [[META6:![0-9]+]] = !{i64 16, !"_ZTS7Derived"}
; GEN: [[META7:![0-9]+]] = !{i64 16, !"_ZTSM7DerivedFiiiE.virtual"}
; GEN: [[META8:![0-9]+]] = !{i64 40, !"_ZTSM7DerivedFiiiE.virtual"}
; GEN: [[META9:![0-9]+]] = !{i64 16, !"_ZTS5Base2"}
;.
; LOWER: [[META0:![0-9]+]] = !{i64 16, !"_ZTS5Base1"}
; LOWER: [[META1:![0-9]+]] = !{i64 16, !"_ZTSM5Base1FiiiE.virtual"}
; LOWER: [[META2:![0-9]+]] = !{i64 40, !"_ZTSM5Base1FiiiE.virtual"}
; LOWER: [[META3:![0-9]+]] = !{i64 40, !"_ZTS5Base2"}
; LOWER: [[META4:![0-9]+]] = !{i64 16, !"_ZTSM5Base2FiiiE.virtual"}
; LOWER: [[META5:![0-9]+]] = !{i64 40, !"_ZTSM5Base2FiiiE.virtual"}
; LOWER: [[META6:![0-9]+]] = !{i64 16, !"_ZTS7Derived"}
; LOWER: [[META7:![0-9]+]] = !{i64 16, !"_ZTSM7DerivedFiiiE.virtual"}
; LOWER: [[META8:![0-9]+]] = !{i64 40, !"_ZTSM7DerivedFiiiE.virtual"}
; LOWER: [[META9:![0-9]+]] = !{i64 16, !"_ZTS5Base2"}
;.
