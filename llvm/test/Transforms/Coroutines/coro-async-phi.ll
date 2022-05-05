; RUN: opt < %s -enable-coroutines -O0 -S | FileCheck --check-prefixes=CHECK %s
; RUN: opt < %s -enable-coroutines -passes='default<O0>' -S | FileCheck --check-prefixes=CHECK %s

%swift.async_func_pointer = type <{ i32, i32 }>
%swift.context = type { %swift.context*, void (%swift.context*)*, i64 }
%T10RR13AC = type <{ %swift.refcounted, %swift.defaultactor }>
%swift.refcounted = type { %swift.type*, i64 }
%swift.type = type { i64 }
%swift.defaultactor = type { [10 x i8*] }
%swift.bridge = type opaque
%swift.error = type opaque
%swift.executor = type {}

@repoTU = hidden global %swift.async_func_pointer <{ i32 trunc (i64 sub (i64 ptrtoint (void (%swift.context*, i64, i64, %T10RR13AC*)* @repo to i64), i64 ptrtoint (%swift.async_func_pointer* @repoTU to i64)) to i32), i32 20 }>, section "__TEXT,__const", align 8

declare void @use(i8*)

; This used to crash.
; CHECK: repo
define hidden swifttailcc void @repo(%swift.context* swiftasync %arg, i64 %arg1, i64 %arg2, %T10RR13AC* swiftself %arg3) #0 {
entry:
  %i = alloca %swift.context*, align 8
  %i11 = call token @llvm.coro.id.async(i32 20, i32 16, i32 0, i8* bitcast (%swift.async_func_pointer* @repoTU to i8*))
  %i12 = call i8* @llvm.coro.begin(token %i11, i8* null)
  %i18 = call i8* @llvm.coro.async.resume()
  call void @use(i8* %i18)
  %i21 = call { i8* } (i32, i8*, i8*, ...) @llvm.coro.suspend.async.sl_p0i8s(i32 0, i8* %i18, i8* bitcast (i8* (i8*)* @__swift_async_resume_get_context to i8*), i8* bitcast (void (i8*, %swift.executor*, %swift.context*)* @__swift_suspend_point to i8*), i8* %i18, %swift.executor* null, %swift.context* null)
  %i22 = extractvalue { i8* } %i21, 0
  %i23 = call i8* @__swift_async_resume_get_context(i8* %i22)
  %i28 = icmp eq i64 %arg2, 0
  br i1 %i28, label %bb126, label %bb

bb:                                               ; preds = %entry
  %i29 = inttoptr i64 %arg2 to %swift.bridge*
  br label %bb30

bb30:                                             ; preds = %bb
  %i31 = phi i64 [ %arg1, %bb ]
  %i32 = phi %swift.bridge* [ %i29, %bb ]
  %i35 = ptrtoint %swift.bridge* %i32 to i64
  %i36 = bitcast %T10RR13AC* %arg3 to %swift.type**
  %i37 = load %swift.type*, %swift.type** %i36, align 8
  %i38 = bitcast %swift.type* %i37 to void (%swift.context*, i64, i64, %T10RR13AC*)**
  %i39 = getelementptr inbounds void (%swift.context*, i64, i64, %T10RR13AC*)*, void (%swift.context*, i64, i64, %T10RR13AC*)** %i38, i64 11
  %i40 = load void (%swift.context*, i64, i64, %T10RR13AC*)*, void (%swift.context*, i64, i64, %T10RR13AC*)** %i39, align 8
  %i41 = bitcast void (%swift.context*, i64, i64, %T10RR13AC*)* %i40 to %swift.async_func_pointer*
  %i42 = getelementptr inbounds %swift.async_func_pointer, %swift.async_func_pointer* %i41, i32 0, i32 0
  %i43 = load i32, i32* %i42, align 8
  %i44 = sext i32 %i43 to i64
  %i45 = ptrtoint i32* %i42 to i64
  %i46 = add i64 %i45, %i44
  %i47 = inttoptr i64 %i46 to i8*
  %i48 = bitcast i8* %i47 to void (%swift.context*, i64, i64, %T10RR13AC*)*
  %i52 = call swiftcc i8* @swift_task_alloc(i64 24) #1
  %i53 = bitcast i8* %i52 to <{ %swift.context*, void (%swift.context*)*, i32 }>*
  %i54 = load %swift.context*, %swift.context** %i, align 8
  %i55 = getelementptr inbounds <{ %swift.context*, void (%swift.context*)*, i32 }>, <{ %swift.context*, void (%swift.context*)*, i32 }>* %i53, i32 0, i32 0
  store %swift.context* %i54, %swift.context** %i55, align 8
  %i56 = call i8* @llvm.coro.async.resume()
  call void @use(i8* %i56)
  %i57 = bitcast i8* %i56 to void (%swift.context*)*
  %i58 = getelementptr inbounds <{ %swift.context*, void (%swift.context*)*, i32 }>, <{ %swift.context*, void (%swift.context*)*, i32 }>* %i53, i32 0, i32 1
  store void (%swift.context*)* %i57, void (%swift.context*)** %i58, align 8
  %i59 = bitcast i8* %i52 to %swift.context*
  %i60 = bitcast void (%swift.context*, i64, i64, %T10RR13AC*)* %i48 to i8*
  %i61 = call { i8*, %swift.error* } (i32, i8*, i8*, ...) @llvm.coro.suspend.async.sl_p0i8p0s_swift.errorss(i32 256, i8* %i56, i8* bitcast (i8* (i8*)* @__swift_async_resume_project_context to i8*), i8* bitcast (void (i8*, %swift.context*, i64, i64, %T10RR13AC*)* @__swift_suspend_dispatch_4 to i8*), i8* %i60, %swift.context* %i59, i64 %i31, i64 0, %T10RR13AC* %arg3)
  %i62 = extractvalue { i8*, %swift.error* } %i61, 0
  %i63 = call i8* @__swift_async_resume_project_context(i8* %i62)
  %i64 = bitcast i8* %i63 to %swift.context*
  store %swift.context* %i64, %swift.context** %i, align 8
  %i65 = extractvalue { i8*, %swift.error* } %i61, 1
  call swiftcc void @swift_task_dealloc(i8* %i52) #1
  br i1 %i28, label %bb126, label %bb68

bb68:                                             ; preds = %bb30
  %i69 = call i8* @llvm.coro.async.resume()
  call void @use(i8* %i69)
  %i70 = load %swift.context*, %swift.context** %i, align 8
  %i71 = call { i8* } (i32, i8*, i8*, ...) @llvm.coro.suspend.async.sl_p0i8s(i32 0, i8* %i69, i8* bitcast (i8* (i8*)* @__swift_async_resume_get_context to i8*), i8* bitcast (void (i8*, %swift.executor*, %swift.context*)* @__swift_suspend_point to i8*), i8* %i69, %swift.executor* null, %swift.context* %i70)
  %i77 = ptrtoint %swift.bridge* %i32 to i64
  %i78 = bitcast %T10RR13AC* %arg3 to %swift.type**
  %i79 = load %swift.type*, %swift.type** %i78, align 8
  %i80 = bitcast %swift.type* %i79 to void (%swift.context*, i64, i64, %T10RR13AC*)**
  %i81 = getelementptr inbounds void (%swift.context*, i64, i64, %T10RR13AC*)*, void (%swift.context*, i64, i64, %T10RR13AC*)** %i80, i64 11
  %i82 = load void (%swift.context*, i64, i64, %T10RR13AC*)*, void (%swift.context*, i64, i64, %T10RR13AC*)** %i81, align 8
  %i83 = bitcast void (%swift.context*, i64, i64, %T10RR13AC*)* %i82 to %swift.async_func_pointer*
  %i84 = getelementptr inbounds %swift.async_func_pointer, %swift.async_func_pointer* %i83, i32 0, i32 0
  %i85 = load i32, i32* %i84, align 8
  %i86 = sext i32 %i85 to i64
  %i87 = ptrtoint i32* %i84 to i64
  %i88 = add i64 %i87, %i86
  %i89 = inttoptr i64 %i88 to i8*
  %i90 = bitcast i8* %i89 to void (%swift.context*, i64, i64, %T10RR13AC*)*
  %i94 = call swiftcc i8* @swift_task_alloc(i64 24) #1
  %i98 = call i8* @llvm.coro.async.resume()
  call void @use(i8* %i98)
  %i102 = bitcast void (%swift.context*, i64, i64, %T10RR13AC*)* %i90 to i8*
  %i103 = call { i8*, %swift.error* } (i32, i8*, i8*, ...) @llvm.coro.suspend.async.sl_p0i8p0s_swift.errorss(i32 256, i8* %i98, i8* bitcast (i8* (i8*)* @__swift_async_resume_project_context to i8*), i8* bitcast (void (i8*, %swift.context*, i64, i64, %T10RR13AC*)* @__swift_suspend_dispatch_4.1 to i8*), i8* %i102, %swift.context* null, i64 %i31, i64 0, %T10RR13AC* %arg3)
  call swiftcc void @swift_task_dealloc(i8* %i94) #1
  br label %bb126

bb126:
  %i162 = call i1 (i8*, i1, ...) @llvm.coro.end.async(i8* %i12, i1 false, void (i8*, %swift.context*, %swift.error*)* @__swift_suspend_dispatch_2, i8* bitcast (void (%swift.context*, %swift.error*)* @doIt to i8*), %swift.context* null, %swift.error* null)
  unreachable
}

; Function Attrs: nounwind
declare token @llvm.coro.id.async(i32, i32, i32, i8*) #1

; Function Attrs: nounwind
declare i8* @llvm.coro.begin(token, i8* writeonly) #1

; Function Attrs: argmemonly nofree nosync nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #3

; Function Attrs: nounwind
declare i8* @llvm.coro.async.resume() #1

; Function Attrs: noinline
define linkonce_odr hidden i8* @__swift_async_resume_get_context(i8* %arg) #4 {
entry:
  ret i8* %arg
}

; Function Attrs: nounwind
declare extern_weak swifttailcc void @swift_task_switch(%swift.context*, i8*, %swift.executor*) #1

; Function Attrs: nounwind
define internal swifttailcc void @__swift_suspend_point(i8* %arg, %swift.executor* %arg1, %swift.context* %arg2) #1 {
entry:
  musttail call swifttailcc void @swift_task_switch(%swift.context* swiftasync %arg2, i8* %arg, %swift.executor* %arg1) #1
  ret void
}

; Function Attrs: nounwind
declare { i8* } @llvm.coro.suspend.async.sl_p0i8s(i32, i8*, i8*, ...) #1

; Function Attrs: nounwind
declare i1 @llvm.coro.end.async(i8*, i1, ...) #1

; Function Attrs: argmemonly nounwind
declare extern_weak swiftcc i8* @swift_task_alloc(i64) #5

; Function Attrs: nounwind readnone
declare i8** @llvm.swift.async.context.addr() #6

; Function Attrs: alwaysinline nounwind
define linkonce_odr hidden i8* @__swift_async_resume_project_context(i8* %arg) #7 {
entry:
  %i = bitcast i8* %arg to i8**
  %i1 = load i8*, i8** %i, align 8
  %i2 = call i8** @llvm.swift.async.context.addr()
  store i8* %i1, i8** %i2, align 8
  ret i8* %i1
}

; Function Attrs: nounwind
declare { i8*, %swift.error* } @llvm.coro.suspend.async.sl_p0i8p0s_swift.errorss(i32, i8*, i8*, ...) #1

; Function Attrs: argmemonly nounwind
declare extern_weak swiftcc void @swift_task_dealloc(i8*) #5

; Function Attrs: nounwind
define internal swifttailcc void @__swift_suspend_dispatch_4(i8* %arg, %swift.context* %arg1, i64 %arg2, i64 %arg3, %T10RR13AC* %arg4) #1 {
entry:
  %i = bitcast i8* %arg to void (%swift.context*, i64, i64, %T10RR13AC*)*
  musttail call swifttailcc void %i(%swift.context* swiftasync %arg1, i64 %arg2, i64 %arg3, %T10RR13AC* swiftself %arg4)
  ret void
}

declare swifttailcc void @doIt(%swift.context* swiftasync %arg1, %swift.error* swiftself %arg2)

; Function Attrs: nounwind
define internal swifttailcc void @__swift_suspend_dispatch_2(i8* %arg, %swift.context* %arg1, %swift.error* %arg2) #1 {
entry:
  %i = bitcast i8* %arg to void (%swift.context*, %swift.error*)*
  musttail call swifttailcc void %i(%swift.context* swiftasync %arg1, %swift.error* swiftself %arg2)
  ret void
}

; Function Attrs: nounwind
define internal swifttailcc void @__swift_suspend_dispatch_4.1(i8* %arg, %swift.context* %arg1, i64 %arg2, i64 %arg3, %T10RR13AC* %arg4) #1 {
entry:
  %i = bitcast i8* %arg to void (%swift.context*, i64, i64, %T10RR13AC*)*
  musttail call swifttailcc void %i(%swift.context* swiftasync %arg1, i64 %arg2, i64 %arg3, %T10RR13AC* swiftself %arg4)
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
