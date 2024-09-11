; RUN: opt -mtriple='arm64-' %s -S -passes='module(coro-early),cgscc(coro-split,simplifycfg)' -o - | FileCheck %s
; RUN: opt -mtriple='x86_64' %s -S -passes='module(coro-early),cgscc(coro-split,simplifycfg)' -o - | FileCheck %s
; RUN: opt -mtriple='i386-' %s -S -passes='module(coro-early),cgscc(coro-split,simplifycfg)' -o - | FileCheck %s --check-prefix=NOENTRY
; RUN: opt -mtriple='armv7-' %s -S -passes='module(coro-early),cgscc(coro-split,simplifycfg)' -o - | FileCheck %s --check-prefix=NOENTRY

;; Replicate those tests with non-instruction debug markers.
; RUN: opt --try-experimental-debuginfo-iterators -mtriple='arm64-' %s -S -passes='module(coro-early),cgscc(coro-split,simplifycfg)' -o - | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators -mtriple='x86_64' %s -S -passes='module(coro-early),cgscc(coro-split,simplifycfg)' -o - | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators -mtriple='i386-' %s -S -passes='module(coro-early),cgscc(coro-split,simplifycfg)' -o - | FileCheck %s --check-prefix=NOENTRY
; RUN: opt --try-experimental-debuginfo-iterators -mtriple='armv7-' %s -S -passes='module(coro-early),cgscc(coro-split,simplifycfg)' -o - | FileCheck %s --check-prefix=NOENTRY

; NOENTRY-NOT: OP_llvm_entry_value

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

; This coroutine has one split point and two variables defined by:
;   %var_with_dbg_value,   which has multiple dbg.value intrinsics associated with
;                          it, one per split point.
;   %var_with_dbg_declare, which has a single dbg.declare intrinsic associated
;                          with it at the coroutine entry.
; We check that, for each funclet, the debug intrinsics are propagated properly AND that
; an `entry_value` operation is created.
define swifttailcc void @coroutineA(ptr swiftasync %arg) !dbg !48 {
  %var_with_dbg_value = alloca ptr, align 8
  %var_with_dbg_declare = alloca ptr, align 8
  call void @llvm.dbg.declare(metadata ptr %var_with_dbg_declare, metadata !500, metadata !DIExpression()), !dbg !54
  call void @llvm.dbg.value(metadata ptr %var_with_dbg_value, metadata !50, metadata !DIExpression(DW_OP_deref)), !dbg !54
  %i2 = call token @llvm.coro.id.async(i32 16, i32 16, i32 0, ptr nonnull @coroutineATu)
  %i3 = call ptr @llvm.coro.begin(token %i2, ptr null)
; CHECK-LABEL: define {{.*}} @coroutineA(
; CHECK-SAME:    ptr swiftasync %[[frame_ptr:.*]])
; CHECK:      #dbg_declare(ptr %[[frame_ptr]], {{.*}} !DIExpression(
; CHECK-SAME:                   DW_OP_plus_uconst, 24)
; CHECK:      #dbg_value(ptr %[[frame_ptr]], {{.*}} !DIExpression(
; CHECK-SAME:                 DW_OP_plus_uconst, 16, DW_OP_deref)
; CHECK:      call {{.*}} @swift_task_switch

  %i7 = call ptr @llvm.coro.async.resume(), !dbg !54
  %i10 = call { ptr } (i32, ptr, ptr, ...) @llvm.coro.suspend.async.sl_p0s(i32 0, ptr %i7, ptr nonnull @__swift_async_resume_get_context, ptr nonnull @coroutineA.1, ptr %i7, i64 0, i64 0, ptr %arg), !dbg !54
  %i11 = extractvalue { ptr } %i10, 0, !dbg !54
  %i12 = call ptr @__swift_async_resume_get_context(ptr %i11), !dbg !54
  call void @dont_optimize(ptr %var_with_dbg_value, ptr %var_with_dbg_declare)
  call void @llvm.dbg.value(metadata ptr %var_with_dbg_value, metadata !50, metadata !DIExpression(DW_OP_deref)), !dbg !54
  %i17 = load i32, ptr getelementptr inbounds (<{i32, i32}>, ptr @coroutineBTu, i64 0, i32 1), align 8, !dbg !54
  call void @llvm.dbg.value(metadata !DIArgList(ptr %var_with_dbg_value, i32 %i17), metadata !501, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_deref)), !dbg !54
  %i18 = zext i32 %i17 to i64, !dbg !54
  %i19 = call swiftcc ptr @swift_task_alloc(i64 %i18), !dbg !54
; CHECK-NOT: define
; CHECK-LABEL: define {{.*}} @coroutineATY0_(
; CHECK-SAME:    ptr swiftasync %[[frame_ptr:.*]])
; CHECK:      #dbg_declare(ptr %[[frame_ptr]], {{.*}} !DIExpression(
; CHECK-SAME:                   DW_OP_LLVM_entry_value, 1, DW_OP_plus_uconst, 24)
; CHECK:      #dbg_value(ptr %[[frame_ptr]], {{.*}} !DIExpression(
; CHECK-SAME:                 DW_OP_LLVM_entry_value, 1, DW_OP_plus_uconst, 16, DW_OP_deref)
; CHECK:      #dbg_value(!DIArgList(ptr %[[frame_ptr]], i32 %{{.*}}), {{.*}} !DIExpression(
; CHECK-SAME:                 DW_OP_LLVM_arg, 0, DW_OP_plus_uconst, 16, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_deref)
; CHECK:      call {{.*}} @coroutineB

  %i23 = call ptr @llvm.coro.async.resume(), !dbg !54
  %i25 = getelementptr inbounds <{ ptr, ptr }>, ptr %i19, i64 0, i32 1, !dbg !54
  store ptr %i23, ptr %i25, align 8, !dbg !54
  %i27 = call { ptr } (i32, ptr, ptr, ...) @llvm.coro.suspend.async.sl_p0s(i32 0, ptr %i23, ptr nonnull @__swift_async_resume_project_context, ptr nonnull @coroutineA.0, ptr nonnull @coroutineB, ptr nonnull %i19), !dbg !54
  %i28 = extractvalue { ptr } %i27, 0, !dbg !54
  %i29 = call ptr @__swift_async_resume_project_context(ptr %i28), !dbg !54
  call swiftcc void @swift_task_dealloc(ptr nonnull %i19), !dbg !54
  call void @dont_optimize(ptr %var_with_dbg_value, ptr %var_with_dbg_declare)
  call void @llvm.dbg.value(metadata ptr %var_with_dbg_value, metadata !50, metadata !DIExpression(DW_OP_deref)), !dbg !54
; CHECK-NOT: define
; CHECK-LABEL: define {{.*}} @coroutineATQ1_(
; CHECK-SAME:    ptr swiftasync %[[frame_ptr:.*]])
; Note the extra level of indirection that shows up here!
; CHECK:      #dbg_declare(ptr %[[frame_ptr]], {{.*}} !DIExpression(
; CHECK-SAME:                   DW_OP_LLVM_entry_value, 1, DW_OP_deref, DW_OP_plus_uconst, 24)
; CHECK:      #dbg_value(ptr %[[frame_ptr]], {{.*}} !DIExpression(
; CHECK-SAME:                 DW_OP_LLVM_entry_value, 1, DW_OP_deref, DW_OP_plus_uconst, 16, DW_OP_deref)
; CHECK:      call {{.*}} @swift_task_switch

  %i31 = call ptr @llvm.coro.async.resume(), !dbg !54
  %i33 = call { ptr } (i32, ptr, ptr, ...) @llvm.coro.suspend.async.sl_p0s(i32 0, ptr %i31, ptr nonnull @__swift_async_resume_get_context, ptr nonnull @coroutineA.1, ptr %i31, i64 0, i64 0, ptr %i29), !dbg !54
  %i34 = extractvalue { ptr } %i33, 0, !dbg !54
  %i35 = call ptr @__swift_async_resume_get_context(ptr %i34), !dbg !54
  %i45 = call i1 (ptr, i1, ...) @llvm.coro.end.async(ptr %i3, i1 false, ptr nonnull @coroutineA.0.1, ptr undef, ptr undef), !dbg !54
  unreachable, !dbg !54
; CHECK-NOT: define
; CHECK-LABEL: define {{.*}} @coroutineATY2_(
; CHECK-SAME:    ptr swiftasync %[[frame_ptr:.*]])
; CHECK:      #dbg_declare(ptr %[[frame_ptr]], {{.*}} !DIExpression(
; CHECK-SAME:                   DW_OP_LLVM_entry_value, 1, DW_OP_plus_uconst, 24)
}

; Everything from here on is just support code for the coroutines.

@coroutineBTu = global <{i32, i32}> <{ i32 trunc (i64 sub (i64 ptrtoint (ptr @"coroutineB" to i64), i64 ptrtoint (ptr @"coroutineBTu" to i64)) to i32), i32 16 }>, align 8
@coroutineATu = global <{i32, i32}> <{ i32 trunc (i64 sub (i64 ptrtoint (ptr @"coroutineA" to i64), i64 ptrtoint (ptr @"coroutineATu" to i64)) to i32), i32 16 }>, align 8

define weak_odr hidden ptr @__swift_async_resume_get_context(ptr %arg) !dbg !64 {
  ret ptr %arg, !dbg !65
}
define hidden swifttailcc void @coroutineA.1(ptr %arg, i64 %arg1, i64 %arg2, ptr %arg3) !dbg !66 {
  musttail call swifttailcc void @swift_task_switch(ptr swiftasync %arg3, ptr %arg, i64 %arg1, i64 %arg2), !dbg !67
  ret void, !dbg !67
}

define weak_odr hidden ptr @__swift_async_resume_project_context(ptr %arg) !dbg !68 {
  %i1 = load ptr, ptr %arg, align 8, !dbg !69
  %i2 = call ptr @llvm.swift.async.context.addr(), !dbg !69
  store ptr %i1, ptr %i2, align 8, !dbg !69
  ret ptr %i1, !dbg !69
}
define hidden swifttailcc void @coroutineA.0(ptr %arg, ptr %arg1) !dbg !70 {
  musttail call swifttailcc void %arg(ptr swiftasync %arg1), !dbg !71
  ret void, !dbg !71
}
define hidden swifttailcc void @coroutineA.0.1(ptr %arg, ptr %arg1) !dbg !72 {
  musttail call swifttailcc void %arg(ptr swiftasync %arg1), !dbg !73
  ret void, !dbg !73
}
define swifttailcc void @coroutineB(ptr swiftasync %arg) !dbg !37 {
  %i2 = call token @llvm.coro.id.async(i32 16, i32 16, i32 0, ptr nonnull @coroutineBTu)
  %i3 = call ptr @llvm.coro.begin(token %i2, ptr null)
  %i6 = getelementptr inbounds <{ ptr, ptr }>, ptr %arg, i64 0, i32 1, !dbg !42
  %i712 = load ptr, ptr %i6, align 8, !dbg !42
  %i10 = call i1 (ptr, i1, ...) @llvm.coro.end.async(ptr %i3, i1 false, ptr nonnull @coroutineB.0, ptr %i712, ptr %arg), !dbg !42
  unreachable, !dbg !42
}
define hidden swifttailcc void @coroutineB.0(ptr %arg, ptr %arg1) !dbg !44 {
  musttail call swifttailcc void %arg(ptr swiftasync %arg1), !dbg !47
  ret void, !dbg !47
}

declare i1 @llvm.coro.end.async(ptr, i1, ...)
declare ptr @llvm.coro.async.resume()
declare ptr @llvm.coro.begin(token, ptr writeonly)
declare ptr @llvm.swift.async.context.addr()
declare swiftcc ptr @swift_task_alloc(i64)
declare swiftcc void @swift_task_dealloc(ptr)
declare swifttailcc void @swift_task_switch(ptr, ptr, i64, i64)
declare token @llvm.coro.id.async(i32, i32, i32, ptr)
declare void @dont_optimize(ptr, ptr)
declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @llvm.dbg.value(metadata, metadata, metadata)
declare { ptr } @llvm.coro.suspend.async.sl_p0s(i32, ptr, ptr, ...)

!llvm.module.flags = !{!6, !7}
!llvm.dbg.cu = !{!16}

!6 = !{i32 7, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}

!50 = !DILocalVariable(name: "k1", scope: !48, file: !17, line: 7, type: !53)
!500 = !DILocalVariable(name: "k2", scope: !48, file: !17, line: 7, type: !53)
!501 = !DILocalVariable(name: "k3", scope: !48, file: !17, line: 7, type: !53)
!49 = !{!50, !500}

!16 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !17, producer: "", emissionKind: FullDebug)
!17 = !DIFile(filename: "blah", directory: "")

!53 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Klass")
!46 = !DISubroutineType(types: null)

!64 = distinct !DISubprogram(linkageName: "blah", file: !17, type: !46, unit: !16)
!68 = distinct !DISubprogram(linkageName: "blah", file: !17, type: !46, unit: !16)
!66 = distinct !DISubprogram(linkageName: "coroutineA", file: !17, type: !46, unit: !16)
!70 = distinct !DISubprogram(linkageName: "coroutineA", file: !17, type: !46, unit: !16)
!72 = distinct !DISubprogram(linkageName: "coroutineA", file: !17, type: !46, unit: !16)
!48 = distinct !DISubprogram(linkageName: "coroutineA", file: !17, type: !46, unit: !16, retainedNodes: !49)
!37 = distinct !DISubprogram(linkageName: "coroutineB", file: !17, type: !46, unit: !16)
!44 = distinct !DISubprogram(linkageName: "coroutineB", file: !17, type: !46, unit: !16)
!65 = !DILocation(line: 0, scope: !64)
!67 = !DILocation(line: 0, scope: !66)
!69 = !DILocation(line: 0, scope: !68)
!71 = !DILocation(line: 0, scope: !70)
!73 = !DILocation(line: 0, scope: !72)
!54 = !DILocation(line: 6, scope: !48)
!42 = !DILocation(line: 3, scope: !37)
!47 = !DILocation(line: 0, scope: !44)
