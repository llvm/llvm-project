; RUN: opt < %s -enable-coroutines -O0 -S | FileCheck --check-prefixes=CHECK %s
; RUN: opt < %s -enable-coroutines -passes='default<O0>' -S | FileCheck --check-prefixes=CHECK %s

%swift.async_func_pointer = type <{ i32, i32 }>
%swift.context = type { %swift.context*, void (%swift.context*)*, i64 }
%T10RR13AC = type <{ %swift.refcounted, %swift.defaultactor }>
%swift.refcounted = type { %swift.type*, i64 }
%swift.type = type { i64 }
%swift.defaultactor = type { [10 x i8*] }
%TSSSg = type <{ [16 x i8] }>
%TSS = type <{ %Ts11_StringGutsV }>
%Ts11_StringGutsV = type <{ %Ts13_StringObjectV }>
%Ts13_StringObjectV = type <{ %Ts6UInt64V, %swift.bridge* }>
%Ts6UInt64V = type <{ i64 }>
%swift.bridge = type opaque
%swift.error = type opaque
%swift.executor = type {}

@repoTU = hidden global %swift.async_func_pointer <{ i32 trunc (i64 sub (i64 ptrtoint (void (%swift.context*, i64, i64, %T10RR13AC*)* @repo to i64), i64 ptrtoint (%swift.async_func_pointer* @repoTU to i64)) to i32), i32 20 }>, section "__TEXT,__const", align 8
@0 = external hidden unnamed_addr constant [28 x i8]
@1 = external hidden unnamed_addr constant [12 x i8]

; This used to crash.
; CHECK: @repo

define hidden swifttailcc void @repo(%swift.context* swiftasync %0, i64 %1, i64 %2, %T10RR13AC* swiftself %3) #0 {
entry:
  %4 = alloca %swift.context*, align 8
  %username.debug = alloca %TSSSg, align 8
  %5 = load %TSSSg, %TSSSg* %username.debug, align 1
  %6 = bitcast %TSSSg* %username.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %6, i8 0, i64 16, i1 false)
  %self.debug = alloca %T10RR13AC*, align 8
  %7 = load %T10RR13AC*, %T10RR13AC** %self.debug, align 8
  %8 = bitcast %T10RR13AC** %self.debug to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %8, i8 0, i64 8, i1 false)
  %username.debug1 = alloca %TSS, align 8
  %9 = load %TSS, %TSS* %username.debug1, align 1
  %10 = bitcast %TSS* %username.debug1 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %10, i8 0, i64 16, i1 false)
  %swifterror = alloca swifterror %swift.error*, align 8
  store %swift.error* null, %swift.error** %swifterror, align 8
  store %swift.context* %0, %swift.context** %4, align 8
  %11 = bitcast %swift.context* %0 to <{ %swift.context*, void (%swift.context*)*, i32 }>*
  %12 = call token @llvm.coro.id.async(i32 20, i32 16, i32 0, i8* bitcast (%swift.async_func_pointer* @repoTU to i8*))
  %13 = call i8* @llvm.coro.begin(token %12, i8* null)
  %14 = bitcast %TSSSg* %username.debug to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %14)
  %15 = bitcast %TSSSg* %username.debug to { i64, i64 }*
  %16 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %15, i32 0, i32 0
  store i64 %1, i64* %16, align 8
  %17 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %15, i32 0, i32 1
  store i64 %2, i64* %17, align 8
  call void asm sideeffect "", "r"(%TSSSg* %username.debug)
  store %T10RR13AC* %3, %T10RR13AC** %self.debug, align 8
  call void asm sideeffect "", "r"(%T10RR13AC** %self.debug)
  %18 = bitcast %T10RR13AC* %3 to %swift.executor*
  %19 = call i8* @llvm.coro.async.resume()
  %20 = load %swift.context*, %swift.context** %4, align 8
  %21 = load %swift.context*, %swift.context** %4, align 8
  %22 = call { i8* } (i32, i8*, i8*, ...) @llvm.coro.suspend.async.sl_p0i8s(i32 0, i8* %19, i8* bitcast (i8* (i8*)* @__swift_async_resume_get_context to i8*), i8* bitcast (void (i8*, %swift.executor*, %swift.context*)* @__swift_suspend_point to i8*), i8* %19, %swift.executor* %18, %swift.context* %21)
  %23 = extractvalue { i8* } %22, 0
  %24 = call i8* @__swift_async_resume_get_context(i8* %23)
  %25 = bitcast i8* %24 to %swift.context*
  store %swift.context* %25, %swift.context** %4, align 8
  %26 = inttoptr i64 %2 to %swift.bridge*
  %27 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %26) #1
  %28 = ptrtoint %swift.bridge* %26 to i64
  %29 = icmp eq i64 %2, 0
  br i1 %29, label %128, label %30

30:                                               ; preds = %entry
  %31 = inttoptr i64 %2 to %swift.bridge*
  br label %32

32:                                               ; preds = %30
  %33 = phi i64 [ %1, %30 ]
  %34 = phi %swift.bridge* [ %31, %30 ]
  %35 = bitcast %TSS* %username.debug1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %35)
  %username.debug1._guts = getelementptr inbounds %TSS, %TSS* %username.debug1, i32 0, i32 0
  %username.debug1._guts._object = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %username.debug1._guts, i32 0, i32 0
  %username.debug1._guts._object._countAndFlagsBits = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %username.debug1._guts._object, i32 0, i32 0
  %username.debug1._guts._object._countAndFlagsBits._value = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %username.debug1._guts._object._countAndFlagsBits, i32 0, i32 0
  store i64 %33, i64* %username.debug1._guts._object._countAndFlagsBits._value, align 8
  %username.debug1._guts._object._object = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %username.debug1._guts._object, i32 0, i32 1
  store %swift.bridge* %34, %swift.bridge** %username.debug1._guts._object._object, align 8
  call void asm sideeffect "", "r"(%TSS* %username.debug1)
  %36 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %34) #1
  %37 = ptrtoint %swift.bridge* %34 to i64
  %38 = bitcast %T10RR13AC* %3 to %swift.type**
  %39 = load %swift.type*, %swift.type** %38, align 8
  %40 = bitcast %swift.type* %39 to void (%swift.context*, i64, i64, %T10RR13AC*)**
  %41 = getelementptr inbounds void (%swift.context*, i64, i64, %T10RR13AC*)*, void (%swift.context*, i64, i64, %T10RR13AC*)** %40, i64 11
  %42 = load void (%swift.context*, i64, i64, %T10RR13AC*)*, void (%swift.context*, i64, i64, %T10RR13AC*)** %41, align 8, !invariant.load !28
  %43 = bitcast void (%swift.context*, i64, i64, %T10RR13AC*)* %42 to %swift.async_func_pointer*
  %44 = getelementptr inbounds %swift.async_func_pointer, %swift.async_func_pointer* %43, i32 0, i32 0
  %45 = load i32, i32* %44, align 8
  %46 = sext i32 %45 to i64
  %47 = ptrtoint i32* %44 to i64
  %48 = add i64 %47, %46
  %49 = inttoptr i64 %48 to i8*
  %50 = bitcast i8* %49 to void (%swift.context*, i64, i64, %T10RR13AC*)*
  %51 = getelementptr inbounds %swift.async_func_pointer, %swift.async_func_pointer* %43, i32 0, i32 1
  %52 = load i32, i32* %51, align 8
  %53 = zext i32 %52 to i64
  %54 = call swiftcc i8* @swift_task_alloc(i64 %53) #1
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %54)
  %55 = bitcast i8* %54 to <{ %swift.context*, void (%swift.context*)*, i32 }>*
  %56 = load %swift.context*, %swift.context** %4, align 8
  %57 = getelementptr inbounds <{ %swift.context*, void (%swift.context*)*, i32 }>, <{ %swift.context*, void (%swift.context*)*, i32 }>* %55, i32 0, i32 0
  store %swift.context* %56, %swift.context** %57, align 8
  %58 = call i8* @llvm.coro.async.resume()
  %59 = bitcast i8* %58 to void (%swift.context*)*
  %60 = getelementptr inbounds <{ %swift.context*, void (%swift.context*)*, i32 }>, <{ %swift.context*, void (%swift.context*)*, i32 }>* %55, i32 0, i32 1
  store void (%swift.context*)* %59, void (%swift.context*)** %60, align 8
  %61 = bitcast i8* %54 to %swift.context*
  %62 = bitcast void (%swift.context*, i64, i64, %T10RR13AC*)* %50 to i8*
  %63 = call { i8*, %swift.error* } (i32, i8*, i8*, ...) @llvm.coro.suspend.async.sl_p0i8p0s_swift.errorss(i32 256, i8* %58, i8* bitcast (i8* (i8*)* @__swift_async_resume_project_context to i8*), i8* bitcast (void (i8*, %swift.context*, i64, i64, %T10RR13AC*)* @__swift_suspend_dispatch_4 to i8*), i8* %62, %swift.context* %61, i64 %33, i64 %37, %T10RR13AC* %3)
  %64 = extractvalue { i8*, %swift.error* } %63, 0
  %65 = call i8* @__swift_async_resume_project_context(i8* %64)
  %66 = bitcast i8* %65 to %swift.context*
  store %swift.context* %66, %swift.context** %4, align 8
  %67 = extractvalue { i8*, %swift.error* } %63, 1
  store %swift.error* %67, %swift.error** %swifterror, align 8
  %68 = load %swift.error*, %swift.error** %swifterror, align 8
  call swiftcc void @swift_task_dealloc(i8* %54) #1
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %54)
  %69 = icmp ne %swift.error* %68, null
  br i1 %69, label %137, label %70

70:                                               ; preds = %32
  %71 = call i8* @llvm.coro.async.resume()
  %72 = load %swift.context*, %swift.context** %4, align 8
  %73 = call { i8* } (i32, i8*, i8*, ...) @llvm.coro.suspend.async.sl_p0i8s(i32 0, i8* %71, i8* bitcast (i8* (i8*)* @__swift_async_resume_get_context to i8*), i8* bitcast (void (i8*, %swift.executor*, %swift.context*)* @__swift_suspend_point to i8*), i8* %71, %swift.executor* %18, %swift.context* %72)
  %74 = extractvalue { i8* } %73, 0
  %75 = call i8* @__swift_async_resume_get_context(i8* %74)
  %76 = bitcast i8* %75 to %swift.context*
  store %swift.context* %76, %swift.context** %4, align 8
  %77 = inttoptr i64 %37 to %swift.bridge*
  call void @swift_bridgeObjectRelease(%swift.bridge* %77) #1
  %78 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %34) #1
  %79 = ptrtoint %swift.bridge* %34 to i64
  %80 = bitcast %T10RR13AC* %3 to %swift.type**
  %81 = load %swift.type*, %swift.type** %80, align 8
  %82 = bitcast %swift.type* %81 to void (%swift.context*, i64, i64, %T10RR13AC*)**
  %83 = getelementptr inbounds void (%swift.context*, i64, i64, %T10RR13AC*)*, void (%swift.context*, i64, i64, %T10RR13AC*)** %82, i64 11
  %84 = load void (%swift.context*, i64, i64, %T10RR13AC*)*, void (%swift.context*, i64, i64, %T10RR13AC*)** %83, align 8, !invariant.load !28
  %85 = bitcast void (%swift.context*, i64, i64, %T10RR13AC*)* %84 to %swift.async_func_pointer*
  %86 = getelementptr inbounds %swift.async_func_pointer, %swift.async_func_pointer* %85, i32 0, i32 0
  %87 = load i32, i32* %86, align 8
  %88 = sext i32 %87 to i64
  %89 = ptrtoint i32* %86 to i64
  %90 = add i64 %89, %88
  %91 = inttoptr i64 %90 to i8*
  %92 = bitcast i8* %91 to void (%swift.context*, i64, i64, %T10RR13AC*)*
  %93 = getelementptr inbounds %swift.async_func_pointer, %swift.async_func_pointer* %85, i32 0, i32 1
  %94 = load i32, i32* %93, align 8
  %95 = zext i32 %94 to i64
  %96 = call swiftcc i8* @swift_task_alloc(i64 %95) #1
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %96)
  %97 = bitcast i8* %96 to <{ %swift.context*, void (%swift.context*)*, i32 }>*
  %98 = load %swift.context*, %swift.context** %4, align 8
  %99 = getelementptr inbounds <{ %swift.context*, void (%swift.context*)*, i32 }>, <{ %swift.context*, void (%swift.context*)*, i32 }>* %97, i32 0, i32 0
  store %swift.context* %98, %swift.context** %99, align 8
  %100 = call i8* @llvm.coro.async.resume()
  %101 = bitcast i8* %100 to void (%swift.context*)*
  %102 = getelementptr inbounds <{ %swift.context*, void (%swift.context*)*, i32 }>, <{ %swift.context*, void (%swift.context*)*, i32 }>* %97, i32 0, i32 1
  store void (%swift.context*)* %101, void (%swift.context*)** %102, align 8
  %103 = bitcast i8* %96 to %swift.context*
  %104 = bitcast void (%swift.context*, i64, i64, %T10RR13AC*)* %92 to i8*
  %105 = call { i8*, %swift.error* } (i32, i8*, i8*, ...) @llvm.coro.suspend.async.sl_p0i8p0s_swift.errorss(i32 256, i8* %100, i8* bitcast (i8* (i8*)* @__swift_async_resume_project_context to i8*), i8* bitcast (void (i8*, %swift.context*, i64, i64, %T10RR13AC*)* @__swift_suspend_dispatch_4.1 to i8*), i8* %104, %swift.context* %103, i64 %33, i64 %79, %T10RR13AC* %3)
  %106 = extractvalue { i8*, %swift.error* } %105, 0
  %107 = call i8* @__swift_async_resume_project_context(i8* %106)
  %108 = bitcast i8* %107 to %swift.context*
  store %swift.context* %108, %swift.context** %4, align 8
  %109 = extractvalue { i8*, %swift.error* } %105, 1
  store %swift.error* %109, %swift.error** %swifterror, align 8
  %110 = load %swift.error*, %swift.error** %swifterror, align 8
  call swiftcc void @swift_task_dealloc(i8* %96) #1
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %96)
  %111 = icmp ne %swift.error* %110, null
  br i1 %111, label %146, label %112

112:                                              ; preds = %70
  %113 = call i8* @llvm.coro.async.resume()
  %114 = load %swift.context*, %swift.context** %4, align 8
  %115 = call { i8* } (i32, i8*, i8*, ...) @llvm.coro.suspend.async.sl_p0i8s(i32 0, i8* %113, i8* bitcast (i8* (i8*)* @__swift_async_resume_get_context to i8*), i8* bitcast (void (i8*, %swift.executor*, %swift.context*)* @__swift_suspend_point to i8*), i8* %113, %swift.executor* %18, %swift.context* %114)
  %116 = extractvalue { i8* } %115, 0
  %117 = call i8* @__swift_async_resume_get_context(i8* %116)
  %118 = bitcast i8* %117 to %swift.context*
  store %swift.context* %118, %swift.context** %4, align 8
  %119 = inttoptr i64 %79 to %swift.bridge*
  call void @swift_bridgeObjectRelease(%swift.bridge* %119) #1
  call void @swift_bridgeObjectRelease(%swift.bridge* %34) #1
  %120 = load %swift.context*, %swift.context** %4, align 8
  %121 = bitcast %swift.context* %120 to <{ %swift.context*, void (%swift.context*)*, i32 }>*
  %122 = getelementptr inbounds <{ %swift.context*, void (%swift.context*)*, i32 }>, <{ %swift.context*, void (%swift.context*)*, i32 }>* %121, i32 0, i32 1
  %123 = load void (%swift.context*)*, void (%swift.context*)** %122, align 8
  %124 = bitcast void (%swift.context*)* %123 to void (%swift.context*, %swift.error*)*
  %125 = load %swift.context*, %swift.context** %4, align 8
  %126 = bitcast void (%swift.context*, %swift.error*)* %124 to i8*
  %127 = call i1 (i8*, i1, ...) @llvm.coro.end.async(i8* %13, i1 false, void (i8*, %swift.context*, %swift.error*)* @__swift_suspend_dispatch_2.2, i8* %126, %swift.context* %125, %swift.error* null)
  unreachable

128:                                              ; preds = %entry
  br label %129

129:                                              ; preds = %128
  br label %130

130:                                              ; preds = %129
  %131 = call swiftcc { i64, %swift.bridge* } bitcast ({ i64, %swift.bridge* } ()* @"$ss10fatalError_4file4lines5NeverOSSyXK_s12StaticStringVSutFfA_SSycfu_" to { i64, %swift.bridge* } (%swift.refcounted*)*)(%swift.refcounted* swiftself null)
  %132 = extractvalue { i64, %swift.bridge* } %131, 0
  %133 = extractvalue { i64, %swift.bridge* } %131, 1
  br label %134

134:                                              ; preds = %130
  br label %135

135:                                              ; preds = %134
  call swiftcc void @"$ss17_assertionFailure__4file4line5flagss5NeverOs12StaticStringV_SSAHSus6UInt32VtF"(i64 ptrtoint ([12 x i8]* @1 to i64), i64 11, i8 2, i64 %132, %swift.bridge* %133, i64 ptrtoint ([28 x i8]* @0 to i64), i64 27, i8 2, i64 3, i32 1)
  br label %coro.end

coro.end:                                         ; preds = %135
  %136 = call i1 (i8*, i1, ...) @llvm.coro.end.async(i8* %13, i1 false)
  unreachable

137:                                              ; preds = %32
  %138 = phi %swift.error* [ %68, %32 ]
  store %swift.error* null, %swift.error** %swifterror, align 8
  %139 = call i8* @llvm.coro.async.resume()
  %140 = load %swift.context*, %swift.context** %4, align 8
  %141 = call { i8* } (i32, i8*, i8*, ...) @llvm.coro.suspend.async.sl_p0i8s(i32 0, i8* %139, i8* bitcast (i8* (i8*)* @__swift_async_resume_get_context to i8*), i8* bitcast (void (i8*, %swift.executor*, %swift.context*)* @__swift_suspend_point to i8*), i8* %139, %swift.executor* %18, %swift.context* %140)
  %142 = extractvalue { i8* } %141, 0
  %143 = call i8* @__swift_async_resume_get_context(i8* %142)
  %144 = bitcast i8* %143 to %swift.context*
  store %swift.context* %144, %swift.context** %4, align 8
  %145 = inttoptr i64 %37 to %swift.bridge*
  call void @swift_bridgeObjectRelease(%swift.bridge* %145) #1
  call void @swift_bridgeObjectRelease(%swift.bridge* %34) #1
  br label %155

146:                                              ; preds = %70
  %147 = phi %swift.error* [ %110, %70 ]
  store %swift.error* null, %swift.error** %swifterror, align 8
  %148 = call i8* @llvm.coro.async.resume()
  %149 = load %swift.context*, %swift.context** %4, align 8
  %150 = call { i8* } (i32, i8*, i8*, ...) @llvm.coro.suspend.async.sl_p0i8s(i32 0, i8* %148, i8* bitcast (i8* (i8*)* @__swift_async_resume_get_context to i8*), i8* bitcast (void (i8*, %swift.executor*, %swift.context*)* @__swift_suspend_point to i8*), i8* %148, %swift.executor* %18, %swift.context* %149)
  %151 = extractvalue { i8* } %150, 0
  %152 = call i8* @__swift_async_resume_get_context(i8* %151)
  %153 = bitcast i8* %152 to %swift.context*
  store %swift.context* %153, %swift.context** %4, align 8
  %154 = inttoptr i64 %79 to %swift.bridge*
  call void @swift_bridgeObjectRelease(%swift.bridge* %154) #1
  call void @swift_bridgeObjectRelease(%swift.bridge* %34) #1
  br label %155

155:                                              ; preds = %146, %137
  %156 = phi %swift.error* [ %138, %137 ], [ %147, %146 ]
  %157 = load %swift.context*, %swift.context** %4, align 8
  %158 = bitcast %swift.context* %157 to <{ %swift.context*, void (%swift.context*)*, i32 }>*
  %159 = getelementptr inbounds <{ %swift.context*, void (%swift.context*)*, i32 }>, <{ %swift.context*, void (%swift.context*)*, i32 }>* %158, i32 0, i32 1
  %160 = load void (%swift.context*)*, void (%swift.context*)** %159, align 8
  %161 = bitcast void (%swift.context*)* %160 to void (%swift.context*, %swift.error*)*
  %162 = load %swift.context*, %swift.context** %4, align 8
  %163 = bitcast void (%swift.context*, %swift.error*)* %161 to i8*
  %164 = call i1 (i8*, i1, ...) @llvm.coro.end.async(i8* %13, i1 false, void (i8*, %swift.context*, %swift.error*)* @__swift_suspend_dispatch_2, i8* %163, %swift.context* %162, %swift.error* %156)
  unreachable
}

; Function Attrs: nounwind
declare token @llvm.coro.id.async(i32, i32, i32, i8*) #1

; Function Attrs: nounwind
declare i8* @llvm.coro.begin(token, i8* writeonly) #1

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #2

; Function Attrs: argmemonly nofree nosync nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #3

; Function Attrs: nounwind
declare i8* @llvm.coro.async.resume() #1


; Function Attrs: nounwind
define linkonce_odr hidden i8* @__swift_async_resume_get_context(i8* %0) #5 {
entry:
  ret i8* %0
}

; Function Attrs: nounwind
declare extern_weak swifttailcc void @swift_task_switch(%swift.context*, i8*, %swift.executor*) #1

; Function Attrs: nounwind
define internal swifttailcc void @__swift_suspend_point(i8* %0, %swift.executor* %1, %swift.context* %2) #1 {
entry:
  musttail call swifttailcc void @swift_task_switch(%swift.context* swiftasync %2, i8* %0, %swift.executor* %1) #1
  ret void
}
; Function Attrs: nounwind
declare { i8* } @llvm.coro.suspend.async.sl_p0i8s(i32, i8*, i8*, ...) #1

; Function Attrs: nounwind
declare %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned) #1

declare hidden swiftcc { i64, %swift.bridge* } @"$ss10fatalError_4file4lines5NeverOSSyXK_s12StaticStringVSutFfA_SSycfu_"() #0

; Function Attrs: noinline
declare swiftcc void @"$ss17_assertionFailure__4file4line5flagss5NeverOs12StaticStringV_SSAHSus6UInt32VtF"(i64, i64, i8, i64, %swift.bridge*, i64, i64, i8, i64, i32) #5

; Function Attrs: nounwind
declare i1 @llvm.coro.end.async(i8*, i1, ...) #1

; Function Attrs: argmemonly nounwind
declare extern_weak swiftcc i8* @swift_task_alloc(i64) #6


declare i8** @llvm.swift.async.context.addr()

; Function Attrs: alwaysinline nounwind
define linkonce_odr hidden i8* @__swift_async_resume_project_context(i8* %0) #7 {
entry:
  %1 = bitcast i8* %0 to i8**
  %2 = load i8*, i8** %1, align 8
  %3 = call i8** @llvm.swift.async.context.addr()
  store i8* %2, i8** %3, align 8
  ret i8* %2
}



; Function Attrs: nounwind
declare { i8*, %swift.error* } @llvm.coro.suspend.async.sl_p0i8p0s_swift.errorss(i32, i8*, i8*, ...) #1

; Function Attrs: argmemonly nounwind
declare extern_weak swiftcc void @swift_task_dealloc(i8*) #6

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #2

; Function Attrs: nounwind
declare void @swift_bridgeObjectRelease(%swift.bridge*) #1

; Function Attrs: nounwind
define internal swifttailcc void @__swift_suspend_dispatch_4(i8* %0, %swift.context* %1, i64 %2, i64 %3, %T10RR13AC* %4) #1 {
entry:
  %5 = bitcast i8* %0 to void (%swift.context*, i64, i64, %T10RR13AC*)*
  musttail call swifttailcc void %5(%swift.context* swiftasync %1, i64 %2, i64 %3, %T10RR13AC* swiftself %4)
  ret void
}
define internal swifttailcc void @__swift_suspend_dispatch_2(i8* %0, %swift.context* %1, %swift.error* %2) #1 {
entry:
  %3 = bitcast i8* %0 to void (%swift.context*, %swift.error*)*
  musttail call swifttailcc void %3(%swift.context* swiftasync %1, %swift.error* swiftself %2)
  ret void
}

; Function Attrs: nounwind
define internal swifttailcc void @__swift_suspend_dispatch_4.1(i8* %0, %swift.context* %1, i64 %2, i64 %3, %T10RR13AC* %4) #1 {
entry:
  %5 = bitcast i8* %0 to void (%swift.context*, i64, i64, %T10RR13AC*)*
  musttail call swifttailcc void %5(%swift.context* swiftasync %1, i64 %2, i64 %3, %T10RR13AC* swiftself %4)
  ret void
}

; Function Attrs: nounwind
define internal swifttailcc void @__swift_suspend_dispatch_2.2(i8* %0, %swift.context* %1, %swift.error* %2) #1 {
entry:
  %3 = bitcast i8* %0 to void (%swift.context*, %swift.error*)*
  musttail call swifttailcc void %3(%swift.context* swiftasync %1, %swift.error* swiftself %2)
  ret void
}


attributes #0 = { "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
attributes #2 = { argmemonly nofree nosync nounwind willreturn }
attributes #3 = { argmemonly nofree nosync nounwind willreturn writeonly }
attributes #4 = { nounwind "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { noinline "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { argmemonly nounwind }
attributes #7 = { alwaysinline nounwind "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }

!swift.module.flags = !{!0}
!llvm.asan.globals = !{!1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12}
!llvm.module.flags = !{!13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23}
!llvm.linker.options = !{!24, !25, !26, !27}

!0 = !{!"standard-library", i1 false}
!1 = !{%swift.async_func_pointer* @repoTU, null, null, i1 false, i1 true}
!2 = distinct !{null, null, null, i1 false, i1 true}
!3 = distinct !{null, null, null, i1 false, i1 true}
!4 = distinct !{null, null, null, i1 false, i1 true}
!5 = distinct !{null, null, null, i1 false, i1 true}
!6 = distinct !{null, null, null, i1 false, i1 true}
!7 = distinct !{null, null, null, i1 false, i1 true}
!8 = distinct !{null, null, null, i1 false, i1 true}
!9 = distinct !{null, null, null, i1 false, i1 true}
!10 = distinct !{null, null, null, i1 false, i1 true}
!11 = distinct !{null, null, null, i1 false, i1 true}
!12 = distinct !{null, null, null, i1 false, i1 true}
!13 = !{i32 1, !"Objective-C Version", i32 2}
!14 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!15 = !{i32 1, !"Objective-C Image Info Section", !"__DATA,__objc_imageinfo,regular,no_dead_strip"}
!16 = !{i32 1, !"Objective-C Garbage Collection", i8 0}
!17 = !{i32 1, !"Objective-C Class Properties", i32 64}
!18 = !{i32 1, !"wchar_size", i32 4}
!19 = !{i32 7, !"PIC Level", i32 2}
!20 = !{i32 1, !"Swift Version", i32 7}
!21 = !{i32 1, !"Swift ABI Version", i32 7}
!22 = !{i32 1, !"Swift Major Version", i8 5}
!23 = !{i32 1, !"Swift Minor Version", i8 5}
!24 = !{!"-lswiftSwiftOnoneSupport"}
!25 = !{!"-lswiftCore"}
!26 = !{!"-lswift_Concurrency"}
!27 = !{!"-lobjc"}
!28 = !{}
