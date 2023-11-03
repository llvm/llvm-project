; RUN: opt -S -enable-aggressive-mergefunc-ignoringconst -passes=mergefunc-ignoring-const %s -o - | FileCheck %s

%4 = type opaque
%10 = type opaque
%"struct.SearchSpec::State" = type { %4* }
%"struct.PointerList" = type { i8*, i8*, i8*, i8*, i8* }
%"struct.DynamicCallback" = type { %10* }

; CHECK: define ptr @invoke_foo(ptr nocapture readonly %.block_descriptor, ptr %stateWrapper)
; CHECK: %1 = {{.*}}call ptr @invoke_foo.Tm
; CHECK: define ptr @invoke_bar(ptr nocapture readonly %.block_descriptor, ptr %stateWrapper) {
; CHECK: %1 = {{.*}}call ptr @invoke_foo.Tm
; CHECK: define {{.*}}.Tm(ptr nocapture readonly %0, ptr %1, ptr %2, ptr %3)

; Function Attrs: minsize optsize ssp uwtable
define i8* @invoke_foo(i8* nocapture readonly %.block_descriptor, i8* %stateWrapper) #1 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %state = alloca %"struct.SearchSpec::State", align 8
  %agg.tmp = alloca %"struct.PointerList", align 8
  %0 = tail call i8* @llvm.objc.retain(i8* %stateWrapper) #2
  %1 = bitcast %"struct.SearchSpec::State"* %state to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1) #2
  %2 = getelementptr inbounds i8, i8* %stateWrapper, i64 16
  %3 = bitcast i8* %2 to %"struct.SearchSpec::State"* (i8*)**
  %4 = load %"struct.SearchSpec::State"* (i8*)*, %"struct.SearchSpec::State"* (i8*)** %3, align 8
  %call.i4 = invoke nonnull align 8 dereferenceable(8) %"struct.SearchSpec::State"* %4(i8* nonnull %stateWrapper) #31
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  %initialText.i.i = getelementptr inbounds %"struct.SearchSpec::State", %"struct.SearchSpec::State"* %state, i64 0, i32 0
  %initialText2.i.i = getelementptr inbounds %"struct.SearchSpec::State", %"struct.SearchSpec::State"* %call.i4, i64 0, i32 0
  %5 = load %4*, %4** %initialText2.i.i, align 8
  %6 = bitcast %4* %5 to i8*
  %7 = tail call i8* @llvm.objc.retain(i8* %6) #2
  store %4* %5, %4** %initialText.i.i, align 8
  %block.capture.addr = getelementptr inbounds i8, i8* %.block_descriptor, i64 32
  %8 = bitcast i8* %block.capture.addr to i8**
  %9 = load i8*, i8** %8, align 8
  invoke void @callee2(%"struct.PointerList"* nonnull sret(%"struct.PointerList") align 8 %agg.tmp, i8* %9, i1 zeroext false) #31
          to label %invoke.cont2 unwind label %lpad1

invoke.cont2:                                     ; preds = %invoke.cont
  %block.capture.addr3 = getelementptr inbounds i8, i8* %.block_descriptor, i64 40
  %10 = bitcast i8* %block.capture.addr3 to %4**
  %agg.tmp6.sroa.3.0..sroa_idx12 = getelementptr inbounds %"struct.PointerList", %"struct.PointerList"* %agg.tmp, i64 0, i32 3
  %agg.tmp6.sroa.3.0.copyload = load i8*, i8** %agg.tmp6.sroa.3.0..sroa_idx12, align 8
  %11 = load %4*, %4** %10, align 8
  invoke void @callee1(%"struct.SearchSpec::State"* nonnull align 8 dereferenceable(8) %state, %4* %11) #31
          to label %invoke.cont4 unwind label %lpad.i

lpad.i:                                           ; preds = %invoke.cont2
  %12 = landingpad { i8*, i32 }
          cleanup
  call void @llvm.objc.release(i8* %agg.tmp6.sroa.3.0.copyload) #2
  %.phi.trans.insert = bitcast %"struct.SearchSpec::State"* %state to i8**
  %.pre = load i8*, i8** %.phi.trans.insert, align 8
  br label %lpad1.body

invoke.cont4:                                     ; preds = %invoke.cont2
  call void @llvm.objc.release(i8* %agg.tmp6.sroa.3.0.copyload) #2
  %13 = load %4*, %4** %initialText.i.i, align 8
  store %4* null, %4** %initialText.i.i, align 8
  %call78 = call fastcc i8* @callee3(%4* %13) #31 [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  call void (...) @llvm.objc.clang.arc.noop.use(i8* %call78) #2
  %14 = bitcast %"struct.SearchSpec::State"* %state to i8**
  %15 = load i8*, i8** %14, align 8
  call void @llvm.objc.release(i8* %15) #2
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1) #2
  call void @llvm.objc.release(i8* nonnull %stateWrapper) #2, !clang.imprecise_release !1
  %16 = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %call78) #2
  ret i8* %call78

lpad:                                             ; preds = %entry
  %17 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup

lpad1:                                            ; preds = %invoke.cont
  %18 = landingpad { i8*, i32 }
          cleanup
  br label %lpad1.body

lpad1.body:                                       ; preds = %lpad1, %lpad.i
  %19 = phi i8* [ %6, %lpad1 ], [ %.pre, %lpad.i ]
  %eh.lpad-body = phi { i8*, i32 } [ %18, %lpad1 ], [ %12, %lpad.i ]
  call void @llvm.objc.release(i8* %19) #2
  br label %ehcleanup

ehcleanup:                                        ; preds = %lpad1.body, %lpad
  %.pn = phi { i8*, i32 } [ %eh.lpad-body, %lpad1.body ], [ %17, %lpad ]
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1) #2
  call void @llvm.objc.release(i8* nonnull %stateWrapper) #2, !clang.imprecise_release !1
  resume { i8*, i32 } %.pn
}

; Function Attrs: minsize optsize ssp uwtable
define i8* @invoke_bar(i8* nocapture readonly %.block_descriptor, i8* %stateWrapper) #1 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %state = alloca %"struct.DynamicCallback", align 8
  %agg.tmp = alloca %"struct.PointerList", align 8
  %0 = tail call i8* @llvm.objc.retain(i8* %stateWrapper) #2
  %1 = bitcast %"struct.DynamicCallback"* %state to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1) #2
  %2 = getelementptr inbounds i8, i8* %stateWrapper, i64 16
  %3 = bitcast i8* %2 to %"struct.DynamicCallback"* (i8*)**
  %4 = load %"struct.DynamicCallback"* (i8*)*, %"struct.DynamicCallback"* (i8*)** %3, align 8
  %call.i4 = invoke nonnull align 8 dereferenceable(8) %"struct.DynamicCallback"* %4(i8* nonnull %stateWrapper) #31
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  %call.i.i = getelementptr inbounds %"struct.DynamicCallback", %"struct.DynamicCallback"* %state, i64 0, i32 0
  %call2.i.i = getelementptr inbounds %"struct.DynamicCallback", %"struct.DynamicCallback"* %call.i4, i64 0, i32 0
  %5 = load %10*, %10** %call2.i.i, align 8
  %6 = bitcast %10* %5 to i8*
  %7 = tail call i8* @llvm.objc.retain(i8* %6) #2
  store %10* %5, %10** %call.i.i, align 8
  %block.capture.addr = getelementptr inbounds i8, i8* %.block_descriptor, i64 32
  %8 = bitcast i8* %block.capture.addr to i8**
  %9 = load i8*, i8** %8, align 8
  invoke void @callee2(%"struct.PointerList"* nonnull sret(%"struct.PointerList") align 8 %agg.tmp, i8* %9, i1 zeroext false) #31
          to label %invoke.cont2 unwind label %lpad1

invoke.cont2:                                     ; preds = %invoke.cont
  %block.capture.addr3 = getelementptr inbounds i8, i8* %.block_descriptor, i64 40
  %10 = bitcast i8* %block.capture.addr3 to %10**
  %agg.tmp6.sroa.3.0..sroa_idx12 = getelementptr inbounds %"struct.PointerList", %"struct.PointerList"* %agg.tmp, i64 0, i32 3
  %agg.tmp6.sroa.3.0.copyload = load i8*, i8** %agg.tmp6.sroa.3.0..sroa_idx12, align 8
  %11 = load %10*, %10** %10, align 8
  invoke void @callee5(%"struct.DynamicCallback"* nonnull align 8 dereferenceable(8) %state, %10* %11) #31
          to label %invoke.cont4 unwind label %lpad.i

lpad.i:                                           ; preds = %invoke.cont2
  %12 = landingpad { i8*, i32 }
          cleanup
  call void @llvm.objc.release(i8* %agg.tmp6.sroa.3.0.copyload) #2
  %.phi.trans.insert = bitcast %"struct.DynamicCallback"* %state to i8**
  %.pre = load i8*, i8** %.phi.trans.insert, align 8
  br label %lpad1.body

invoke.cont4:                                     ; preds = %invoke.cont2
  call void @llvm.objc.release(i8* %agg.tmp6.sroa.3.0.copyload) #2
  %13 = load %10*, %10** %call.i.i, align 8
  store %10* null, %10** %call.i.i, align 8
  %call78 = call fastcc i8* @callee4(%10* %13) #31 [ "clang.arc.attachedcall"(i8* (i8*)* @llvm.objc.retainAutoreleasedReturnValue) ]
  call void (...) @llvm.objc.clang.arc.noop.use(i8* %call78) #2
  %14 = bitcast %"struct.DynamicCallback"* %state to i8**
  %15 = load i8*, i8** %14, align 8
  call void @llvm.objc.release(i8* %15) #2
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1) #2
  call void @llvm.objc.release(i8* nonnull %stateWrapper) #2, !clang.imprecise_release !1
  %16 = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* %call78) #2
  ret i8* %call78

lpad:                                             ; preds = %entry
  %17 = landingpad { i8*, i32 }
          cleanup
  br label %ehcleanup

lpad1:                                            ; preds = %invoke.cont
  %18 = landingpad { i8*, i32 }
          cleanup
  br label %lpad1.body

lpad1.body:                                       ; preds = %lpad1, %lpad.i
  %19 = phi i8* [ %6, %lpad1 ], [ %.pre, %lpad.i ]
  %eh.lpad-body = phi { i8*, i32 } [ %18, %lpad1 ], [ %12, %lpad.i ]
  call void @llvm.objc.release(i8* %19) #2
  br label %ehcleanup

ehcleanup:                                        ; preds = %lpad1.body, %lpad
  %.pn = phi { i8*, i32 } [ %eh.lpad-body, %lpad1.body ], [ %17, %lpad ]
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1) #2
  call void @llvm.objc.release(i8* nonnull %stateWrapper) #2, !clang.imprecise_release !1
  resume { i8*, i32 } %.pn
}
declare void @callee1(%"struct.SearchSpec::State"* nonnull align 8 dereferenceable(8), %4*)
declare void @callee2(%"struct.PointerList"* sret(%"struct.PointerList") align 8, i8*, i1 zeroext)
declare i8* @callee3(%4* %state.coerce)
declare i8* @callee4(%10* %state.coerce)
declare void @callee5(%"struct.DynamicCallback"* nonnull align 8 dereferenceable(8), %10*)
declare i32 @__gxx_personality_v0(...)
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)
declare i8* @llvm.objc.autoreleaseReturnValue(i8*)
declare void @llvm.objc.clang.arc.noop.use(...)
declare void @llvm.objc.release(i8*)
declare i8* @llvm.objc.retain(i8*)
declare i8* @llvm.objc.retainAutoreleasedReturnValue(i8*)

!1 = !{}
