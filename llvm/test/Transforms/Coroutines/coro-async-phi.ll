; RUN: opt < %s -O0 -S | FileCheck --check-prefixes=CHECK %s

%swift.async_func_pointer = type <{ i32, i32 }>
%swift.context = type { ptr, ptr, i64 }
%T10RR13AC = type <{ %swift.refcounted, %swift.defaultactor }>
%swift.refcounted = type { ptr, i64 }
%swift.type = type { i64 }
%swift.defaultactor = type { [10 x ptr] }
%swift.bridge = type opaque
%swift.error = type opaque
%swift.executor = type {}

@repoTU = hidden global %swift.async_func_pointer <{ i32 trunc (i64 sub (i64 ptrtoint (ptr @repo to i64), i64 ptrtoint (ptr @repoTU to i64)) to i32), i32 20 }>, section "__TEXT,__const", align 8

declare void @use(ptr)

; This used to crash.
; CHECK: repo
define hidden swifttailcc void @repo(ptr swiftasync %arg, i64 %arg1, i64 %arg2, ptr swiftself %arg3) #0 {
entry:
  %i = alloca ptr, align 8
  %i11 = call token @llvm.coro.id.async(i32 20, i32 16, i32 0, ptr @repoTU)
  %i12 = call ptr @llvm.coro.begin(token %i11, ptr null)
  %i18 = call ptr @llvm.coro.async.resume()
  call void @use(ptr %i18)
  %i21 = call { ptr } (i32, ptr, ptr, ...) @llvm.coro.suspend.async.sl_p0i8s(i32 0, ptr %i18, ptr @__swift_async_resume_get_context, ptr @__swift_suspend_point, ptr %i18, ptr null, ptr null)
  %i22 = extractvalue { ptr } %i21, 0
  %i23 = call ptr @__swift_async_resume_get_context(ptr %i22)
  %i28 = icmp eq i64 %arg2, 0
  br i1 %i28, label %bb126, label %bb

bb:                                               ; preds = %entry
  %i29 = inttoptr i64 %arg2 to ptr
  br label %bb30

bb30:                                             ; preds = %bb
  %i31 = phi i64 [ %arg1, %bb ]
  %i32 = phi ptr [ %i29, %bb ]
  %i35 = ptrtoint ptr %i32 to i64
  %i37 = load ptr, ptr %arg3, align 8
  %i39 = getelementptr inbounds ptr, ptr %i37, i64 11
  %i40 = load ptr, ptr %i39, align 8
  %i43 = load i32, ptr %i40, align 8
  %i44 = sext i32 %i43 to i64
  %i45 = ptrtoint ptr %i40 to i64
  %i46 = add i64 %i45, %i44
  %i47 = inttoptr i64 %i46 to ptr
  %i52 = call swiftcc ptr @swift_task_alloc(i64 24) #1
  %i54 = load ptr, ptr %i, align 8
  %i55 = getelementptr inbounds <{ ptr, ptr, i32 }>, <{ ptr, ptr, i32 }>* %i52, i32 0, i32 0
  store ptr %i54, ptr %i55, align 8
  %i56 = call ptr @llvm.coro.async.resume()
  call void @use(ptr %i56)
  %i58 = getelementptr inbounds <{ ptr, ptr, i32 }>, <{ ptr, ptr, i32 }>* %i52, i32 0, i32 1
  store ptr %i56, ptr %i58, align 8
  %i61 = call { ptr, ptr } (i32, ptr, ptr, ...) @llvm.coro.suspend.async.sl_p0i8p0s_swift.errorss(i32 256, ptr %i56, ptr @__swift_async_resume_project_context, ptr @__swift_suspend_dispatch_4, ptr %i47, ptr %i52, i64 %i31, i64 0, ptr %arg3)
  %i62 = extractvalue { ptr, ptr } %i61, 0
  %i63 = call ptr @__swift_async_resume_project_context(ptr %i62)
  store ptr %i63, ptr %i, align 8
  %i65 = extractvalue { ptr, ptr } %i61, 1
  call swiftcc void @swift_task_dealloc(ptr %i52) #1
  br i1 %i28, label %bb126, label %bb68

bb68:                                             ; preds = %bb30
  %i69 = call ptr @llvm.coro.async.resume()
  call void @use(ptr %i69)
  %i70 = load ptr, ptr %i, align 8
  %i71 = call { ptr } (i32, ptr, ptr, ...) @llvm.coro.suspend.async.sl_p0i8s(i32 0, ptr %i69, ptr @__swift_async_resume_get_context, ptr @__swift_suspend_point, ptr %i69, ptr null, ptr %i70)
  %i77 = ptrtoint ptr %i32 to i64
  %i79 = load ptr, ptr %arg3, align 8
  %i81 = getelementptr inbounds ptr, ptr %i79, i64 11
  %i82 = load ptr, ptr %i81, align 8
  %i85 = load i32, ptr %i82, align 8
  %i86 = sext i32 %i85 to i64
  %i87 = ptrtoint ptr %i82 to i64
  %i88 = add i64 %i87, %i86
  %i89 = inttoptr i64 %i88 to ptr
  %i94 = call swiftcc ptr @swift_task_alloc(i64 24) #1
  %i98 = call ptr @llvm.coro.async.resume()
  call void @use(ptr %i98)
  %i103 = call { ptr, ptr } (i32, ptr, ptr, ...) @llvm.coro.suspend.async.sl_p0i8p0s_swift.errorss(i32 256, ptr %i98, ptr @__swift_async_resume_project_context, ptr @__swift_suspend_dispatch_4.1, ptr %i89, ptr null, i64 %i31, i64 0, ptr %arg3)
  call swiftcc void @swift_task_dealloc(ptr %i94) #1
  br label %bb126

bb126:
  %i162 = call i1 (ptr, i1, ...) @llvm.coro.end.async(ptr %i12, i1 false, ptr @__swift_suspend_dispatch_2, ptr @doIt, ptr null, ptr null)
  unreachable
}

; Function Attrs: nounwind
declare token @llvm.coro.id.async(i32, i32, i32, ptr) #1

; Function Attrs: nounwind
declare ptr @llvm.coro.begin(token, ptr writeonly) #1

; Function Attrs: argmemonly nofree nosync nounwind willreturn writeonly
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #3

; Function Attrs: nounwind
declare ptr @llvm.coro.async.resume() #1

; Function Attrs: noinline
define linkonce_odr hidden ptr @__swift_async_resume_get_context(ptr %arg) #4 {
entry:
  ret ptr %arg
}

; Function Attrs: nounwind
declare extern_weak swifttailcc void @swift_task_switch(ptr, ptr, ptr) #1

; Function Attrs: nounwind
define internal swifttailcc void @__swift_suspend_point(ptr %arg, ptr %arg1, ptr %arg2) #1 {
entry:
  musttail call swifttailcc void @swift_task_switch(ptr swiftasync %arg2, ptr %arg, ptr %arg1) #1
  ret void
}

; Function Attrs: nounwind
declare { ptr } @llvm.coro.suspend.async.sl_p0i8s(i32, ptr, ptr, ...) #1

; Function Attrs: nounwind
declare i1 @llvm.coro.end.async(ptr, i1, ...) #1

; Function Attrs: argmemonly nounwind
declare extern_weak swiftcc ptr @swift_task_alloc(i64) #5

; Function Attrs: nounwind readnone
declare ptr @llvm.swift.async.context.addr() #6

; Function Attrs: alwaysinline nounwind
define linkonce_odr hidden ptr @__swift_async_resume_project_context(ptr %arg) #7 {
entry:
  %i1 = load ptr, ptr %arg, align 8
  %i2 = call ptr @llvm.swift.async.context.addr()
  store ptr %i1, ptr %i2, align 8
  ret ptr %i1
}

; Function Attrs: nounwind
declare { ptr, ptr } @llvm.coro.suspend.async.sl_p0i8p0s_swift.errorss(i32, ptr, ptr, ...) #1

; Function Attrs: argmemonly nounwind
declare extern_weak swiftcc void @swift_task_dealloc(ptr) #5

; Function Attrs: nounwind
define internal swifttailcc void @__swift_suspend_dispatch_4(ptr %arg, ptr %arg1, i64 %arg2, i64 %arg3, ptr %arg4) #1 {
entry:
  musttail call swifttailcc void %arg(ptr swiftasync %arg1, i64 %arg2, i64 %arg3, ptr swiftself %arg4)
  ret void
}

declare swifttailcc void @doIt(ptr swiftasync %arg1, ptr swiftself %arg2)

; Function Attrs: nounwind
define internal swifttailcc void @__swift_suspend_dispatch_2(ptr %arg, ptr %arg1, ptr %arg2) #1 {
entry:
  musttail call swifttailcc void %arg(ptr swiftasync %arg1, ptr swiftself %arg2)
  ret void
}

; Function Attrs: nounwind
define internal swifttailcc void @__swift_suspend_dispatch_4.1(ptr %arg, ptr %arg1, i64 %arg2, i64 %arg3, ptr %arg4) #1 {
entry:
  musttail call swifttailcc void %arg(ptr swiftasync %arg1, i64 %arg2, i64 %arg3, ptr swiftself %arg4)
  ret void
}

attributes #0 = { "frame-pointer"="all" }
attributes #1 = { nounwind }
attributes #2 = { argmemonly nofree nosync nounwind willreturn }
attributes #3 = { argmemonly nofree nosync nounwind willreturn writeonly }
attributes #4 = { noinline "frame-pointer"="all" }
attributes #5 = { argmemonly nounwind }
attributes #6 = { nounwind readnone }
attributes #7 = { alwaysinline nounwind "frame-pointer"="all" }
