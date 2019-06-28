; RUN: opt < %s -tapir2target -tapir-target=cilk -debug-abi-calls -S | FileCheck %s
; RUN: opt < %s -passes=tapir2target -tapir-target=cilk -debug-abi-calls -S | FileCheck %s

%class.Foo = type { i8 }

$_ZN3FooC2Ev = comdat any

$__clang_call_terminate = comdat any

@_ZTIi = external constant i8*

; Function Attrs: alwaysinline uwtable
define i32 @_Z3fooP3Foo(%class.Foo* %f) local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %call = invoke i32 @_Z3barP3Foo(%class.Foo* %f)
          to label %try.cont unwind label %lpad

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIi to i8*)
  %1 = extractvalue { i8*, i32 } %0, 0
  %2 = extractvalue { i8*, i32 } %0, 1
  %3 = tail call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #11
  %matches = icmp eq i32 %2, %3
  br i1 %matches, label %catch, label %eh.resume

catch:                                            ; preds = %lpad
  %4 = tail call i8* @__cxa_begin_catch(i8* %1) #11
  %5 = bitcast i8* %4 to i32*
  %6 = load i32, i32* %5, align 4, !tbaa !2
  invoke void @_Z10handle_exni(i32 %6)
          to label %invoke.cont2 unwind label %lpad1

invoke.cont2:                                     ; preds = %catch
  tail call void @__cxa_end_catch() #11
  br label %try.cont

try.cont:                                         ; preds = %entry, %invoke.cont2
  ret i32 0

lpad1:                                            ; preds = %catch
  %7 = landingpad { i8*, i32 }
          cleanup
  %8 = extractvalue { i8*, i32 } %7, 0
  %9 = extractvalue { i8*, i32 } %7, 1
  tail call void @__cxa_end_catch() #11
  br label %eh.resume

eh.resume:                                        ; preds = %lpad1, %lpad
  %ehselector.slot.0 = phi i32 [ %9, %lpad1 ], [ %2, %lpad ]
  %exn.slot.0 = phi i8* [ %8, %lpad1 ], [ %1, %lpad ]
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn.slot.0, 0
  %lpad.val5 = insertvalue { i8*, i32 } %lpad.val, i32 %ehselector.slot.0, 1
  resume { i8*, i32 } %lpad.val5
}

declare i32 @_Z3barP3Foo(%class.Foo*) local_unnamed_addr #1

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #2

declare i8* @__cxa_begin_catch(i8*) local_unnamed_addr

declare void @_Z10handle_exni(i32) local_unnamed_addr #1

declare void @__cxa_end_catch() local_unnamed_addr

; Function Attrs: nounwind
declare i32 @_Z4quuzi(i32) local_unnamed_addr #4

; Function Attrs: nobuiltin
declare noalias nonnull i8* @_Znwm(i64) local_unnamed_addr #6

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr void @_ZN3FooC2Ev(%class.Foo* %this) unnamed_addr #7 comdat align 2 {
entry:
  ret void
}

; Function Attrs: noinline noreturn nounwind
define linkonce_odr hidden void @__clang_call_terminate(i8*) local_unnamed_addr #8 comdat {
  %2 = tail call i8* @__cxa_begin_catch(i8* %0) #11
  tail call void @_ZSt9terminatev() #13
  unreachable
}

declare void @_ZSt9terminatev() local_unnamed_addr

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #9

; Function Attrs: argmemonly
declare void @llvm.detached.rethrow.sl_p0i8i32s(token, { i8*, i32 }) #10

; Function Attrs: uwtable
define void @_Z12spawn_excepti(i32 %n) local_unnamed_addr #5 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-LABEL: define void @_Z12spawn_excepti(i32 %n)
; CHECK: %[[CILKSF:.+]] = alloca %struct.__cilkrts_stack_frame
; CHECK: call void @__cilkrts_enter_frame_1(%struct.__cilkrts_stack_frame* %[[CILKSF]])
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %call = tail call i8* @_Znwm(i64 1) #12
  %0 = bitcast i8* %call to %class.Foo*
  tail call void @_ZN3FooC2Ev(%class.Foo* nonnull %0)
  detach within %syncreg, label %det.achd, label %det.cont unwind label %lpad6
; CHECK: %[[SETJMPRET:.+]] = call i32 @llvm.eh.sjlj.setjmp(
; CHECK: %[[SETJMPBOOL:.+]] = icmp eq i32 %[[SETJMPRET]], 0
; CHECK: br i1 %[[SETJMPBOOL]], label %[[CALLHELPER:.+]], label %[[CONTINUE:.+]]

det.achd:                                         ; preds = %entry
  %call5 = invoke i32 @_Z3barP3Foo(%class.Foo* nonnull %0)
          to label %invoke.cont4 unwind label %lpad1
; CHECK: [[CALLHELPER]]:
; CHECK-NEXT: invoke fastcc void @_Z12spawn_excepti.outline_det.achd.otd1(%class.Foo*
; CHECK-NEXT: to label %[[CONTINUE]] unwind label %lpad6

invoke.cont4:                                     ; preds = %det.achd
  reattach within %syncreg, label %det.cont

det.cont:                                         ; preds = %entry, %invoke.cont4
  %call8 = tail call i32 @_Z4quuzi(i32 %n) #11
  sync within %syncreg, label %sync.continue9

sync.continue9:                                   ; preds = %det.cont
  ret void

lpad1:                                            ; preds = %det.achd
  %1 = landingpad { i8*, i32 }
          catch i8* null
  invoke void @llvm.detached.rethrow.sl_p0i8i32s(token %syncreg, { i8*, i32 } %1)
          to label %det.rethrow.unreachable unwind label %lpad6

det.rethrow.unreachable:                          ; preds = %lpad1
  unreachable

lpad6:                                            ; preds = %entry, %lpad1
  %2 = landingpad { i8*, i32 }
          cleanup
  sync within %syncreg, label %eh.resume
; CHECK: {{^lpad6}}:
; CHECK-NEXT: landingpad { i8*, i32 }
; CHECK-NEXT: cleanup
; CHECK-NEXT: extractvalue
; CHECK-NEXT: call i8* @__cilk_catch_exception(%struct.__cilkrts_stack_frame* %[[CILKSF]]
; CHECK-NEXT: insertvalue
; CHECK-NEXT: call void @__cilk_sync(%struct.__cilkrts_stack_frame* %[[CILKSF]])
; CHECK-NEXT: br label %[[EHRESUME:.+]]
; CHECK: [[EHRESUME]]:
; CHECK-NEXT: call void @__cilk_parent_epilogue(%struct.__cilkrts_stack_frame* %[[CILKSF]])
; CHECK-NEXT: resume { i8*, i32 } %{{.+}}

eh.resume:                                        ; preds = %lpad6
  resume { i8*, i32 } %2
}

; Function Attrs: uwtable
define void @_Z18spawn_throw_inlinei(i32 %n) local_unnamed_addr #5 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-LABEL: define void @_Z18spawn_throw_inlinei(i32 %n)
; CHECK: %[[CILKSF:.+]] = alloca %struct.__cilkrts_stack_frame
; CHECK: call void @__cilkrts_enter_frame_1(%struct.__cilkrts_stack_frame* %[[CILKSF]])
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %call = tail call i8* @_Znwm(i64 1) #12
  %0 = bitcast i8* %call to %class.Foo*
  tail call void @_ZN3FooC2Ev(%class.Foo* nonnull %0)
  detach within %syncreg, label %det.achd, label %det.cont unwind label %lpad6
; CHECK: %[[SETJMPRET:.+]] = call i32 @llvm.eh.sjlj.setjmp(
; CHECK: %[[SETJMPBOOL:.+]] = icmp eq i32 %[[SETJMPRET]], 0
; CHECK: br i1 %[[SETJMPBOOL]], label %[[CALLHELPER:.+]], label %[[CONTINUE:.+]]

det.achd:                                         ; preds = %entry
  %call.i = invoke i32 @_Z3barP3Foo(%class.Foo* nonnull %0)
          to label %invoke.cont4 unwind label %lpad.i
; CHECK: [[CALLHELPER]]:
; CHECK-NEXT: invoke fastcc void @_Z18spawn_throw_inlinei.outline_det.achd.otd1(%class.Foo*
; CHECK-NEXT: to label %[[CONTINUE]] unwind label %lpad6

lpad.i:                                           ; preds = %det.achd
  %1 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIi to i8*)
          catch i8* null
  %2 = extractvalue { i8*, i32 } %1, 0
  %3 = extractvalue { i8*, i32 } %1, 1
  %4 = tail call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #11
  %matches.i = icmp eq i32 %3, %4
  br i1 %matches.i, label %catch.i, label %lpad1.body

catch.i:                                          ; preds = %lpad.i
  %5 = tail call i8* @__cxa_begin_catch(i8* %2) #11
  %6 = bitcast i8* %5 to i32*
  %7 = load i32, i32* %6, align 4, !tbaa !2
  invoke void @_Z10handle_exni(i32 %7)
          to label %invoke.cont2.i unwind label %lpad1.i

invoke.cont2.i:                                   ; preds = %catch.i
  tail call void @__cxa_end_catch() #11
  br label %invoke.cont4

lpad1.i:                                          ; preds = %catch.i
  %8 = landingpad { i8*, i32 }
          catch i8* null
  %9 = extractvalue { i8*, i32 } %8, 0
  %10 = extractvalue { i8*, i32 } %8, 1
  tail call void @__cxa_end_catch() #11
  br label %lpad1.body

invoke.cont4:                                     ; preds = %invoke.cont2.i, %det.achd
  reattach within %syncreg, label %det.cont

det.cont:                                         ; preds = %entry, %invoke.cont4
  %call8 = tail call i32 @_Z4quuzi(i32 %n) #11
  sync within %syncreg, label %sync.continue9

sync.continue9:                                   ; preds = %det.cont
  ret void

lpad1.body:                                       ; preds = %lpad.i, %lpad1.i
  %ehselector.slot.0.i = phi i32 [ %10, %lpad1.i ], [ %3, %lpad.i ]
  %exn.slot.0.i = phi i8* [ %9, %lpad1.i ], [ %2, %lpad.i ]
  %lpad.val.i = insertvalue { i8*, i32 } undef, i8* %exn.slot.0.i, 0
  %lpad.val5.i = insertvalue { i8*, i32 } %lpad.val.i, i32 %ehselector.slot.0.i, 1
  invoke void @llvm.detached.rethrow.sl_p0i8i32s(token %syncreg, { i8*, i32 } %lpad.val5.i)
          to label %det.rethrow.unreachable unwind label %lpad6

det.rethrow.unreachable:                          ; preds = %lpad1.body
  unreachable

lpad6:                                            ; preds = %entry, %lpad1.body
  %11 = landingpad { i8*, i32 }
          cleanup
  sync within %syncreg, label %eh.resume
; CHECK: {{^lpad6}}:
; CHECK-NEXT: landingpad { i8*, i32 }
; CHECK-NEXT: cleanup
; CHECK-NEXT: extractvalue
; CHECK-NEXT: call i8* @__cilk_catch_exception(%struct.__cilkrts_stack_frame* %[[CILKSF]]
; CHECK-NEXT: insertvalue
; CHECK-NEXT: call void @__cilk_sync(%struct.__cilkrts_stack_frame* %[[CILKSF]])
; CHECK-NEXT: br label %[[EHRESUME:.+]]
; CHECK: [[EHRESUME]]:
; CHECK-NEXT: call void @__cilk_parent_epilogue(%struct.__cilkrts_stack_frame* %[[CILKSF]])
; CHECK-NEXT: resume { i8*, i32 } %{{.+}}

eh.resume:                                        ; preds = %lpad6
  resume { i8*, i32 } %11
}

; Function Attrs: uwtable
define void @_Z14spawn_tryblocki(i32 %n) local_unnamed_addr #5 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-LABEL: define void @_Z14spawn_tryblocki(i32 %n)
; CHECK: %[[CILKSF:.+]] = alloca %struct.__cilkrts_stack_frame
; CHECK: call void @__cilkrts_enter_frame_1(%struct.__cilkrts_stack_frame* %[[CILKSF]])
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  detach within %syncreg, label %det.achd, label %det.cont
; CHECK: %[[SETJMPRET:.+]] = call i32 @llvm.eh.sjlj.setjmp(
; CHECK: %[[SETJMPBOOL:.+]] = icmp eq i32 %[[SETJMPRET]], 0
; CHECK: br i1 %[[SETJMPBOOL]], label %[[CALLHELPER1:.+]], label %[[CONTINUE1:.+]]

det.achd:                                         ; preds = %entry
  %call = tail call i32 @_Z4quuzi(i32 %n) #11
  reattach within %syncreg, label %det.cont
; CHECK: [[CALLHELPER1]]:
; CHECK-NEXT: call fastcc void @_Z14spawn_tryblocki.outline_det.achd.otd1(i32 %n)
; CHECK-NEXT: br label %[[CONTINUE1]]

det.cont:                                         ; preds = %det.achd, %entry
  %call1 = invoke i8* @_Znwm(i64 1) #12
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %det.cont
  %0 = bitcast i8* %call1 to %class.Foo*
  tail call void @_ZN3FooC2Ev(%class.Foo* nonnull %0)
  detach within %syncreg, label %det.achd4, label %det.cont10 unwind label %lpad11
; CHECK: %[[SETJMPRET:.+]] = call i32 @llvm.eh.sjlj.setjmp(
; CHECK: %[[SETJMPBOOL:.+]] = icmp eq i32 %[[SETJMPRET]], 0
; CHECK: br i1 %[[SETJMPBOOL]], label %[[CALLHELPER2:.+]], label %[[CONTINUE2:.+]]

det.achd4:                                        ; preds = %invoke.cont
  %call9 = invoke i32 @_Z3barP3Foo(%class.Foo* nonnull %0)
          to label %invoke.cont8 unwind label %lpad5
; CHECK: [[CALLHELPER2]]:
; CHECK-NEXT: invoke fastcc void @_Z14spawn_tryblocki.outline_det.achd4.otd1(%class.Foo*
; CHECK-NEXT: to label %[[CONTINUE2]] unwind label %lpad11

invoke.cont8:                                     ; preds = %det.achd4
  reattach within %syncreg, label %det.cont10

det.cont10:                                       ; preds = %invoke.cont, %invoke.cont8
  detach within %syncreg, label %det.achd13, label %det.cont15
; CHECK: %[[SETJMPRET:.+]] = call i32 @llvm.eh.sjlj.setjmp(
; CHECK: %[[SETJMPBOOL:.+]] = icmp eq i32 %[[SETJMPRET]], 0
; CHECK: br i1 %[[SETJMPBOOL]], label %[[CALLHELPER3:.+]], label %[[CONTINUE3:.+]]

det.achd13:                                       ; preds = %det.cont10
  %call14 = tail call i32 @_Z4quuzi(i32 %n) #11
  reattach within %syncreg, label %det.cont15
; CHECK: [[CALLHELPER3]]:
; CHECK-NEXT: call fastcc void @_Z14spawn_tryblocki.outline_det.achd13.otd1(i32 %n)
; CHECK-NEXT: br label %[[CONTINUE3]]

det.cont15:                                       ; preds = %det.achd13, %det.cont10
  %call18 = invoke i8* @_Znwm(i64 1) #12
          to label %invoke.cont17 unwind label %lpad16

invoke.cont17:                                    ; preds = %det.cont15
  %1 = bitcast i8* %call18 to %class.Foo*
  tail call void @_ZN3FooC2Ev(%class.Foo* nonnull %1)
  %call22 = invoke i32 @_Z3barP3Foo(%class.Foo* nonnull %1)
          to label %invoke.cont21 unwind label %lpad16

invoke.cont21:                                    ; preds = %invoke.cont17
  sync within %syncreg, label %try.cont

lpad:                                             ; preds = %det.cont
  %2 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIi to i8*)
          catch i8* null
  %3 = extractvalue { i8*, i32 } %2, 0
  %4 = extractvalue { i8*, i32 } %2, 1
  br label %ehcleanup26

lpad5:                                            ; preds = %det.achd4
  %5 = landingpad { i8*, i32 }
          catch i8* null
  invoke void @llvm.detached.rethrow.sl_p0i8i32s(token %syncreg, { i8*, i32 } %5)
          to label %det.rethrow.unreachable unwind label %lpad11

det.rethrow.unreachable:                          ; preds = %lpad5
  unreachable

lpad11:                                           ; preds = %invoke.cont, %lpad5
  %6 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIi to i8*)
          catch i8* null
  %7 = extractvalue { i8*, i32 } %6, 0
  %8 = extractvalue { i8*, i32 } %6, 1
  br label %ehcleanup24
; CHECK: {{^lpad11}}:
; CHECK-NEXT: landingpad { i8*, i32 }
; CHECK-NEXT: catch i8* bitcast (i8** @_ZTIi to i8*)
; CHECK-NEXT: catch i8* null

lpad16:                                           ; preds = %invoke.cont17, %det.cont15
  %9 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIi to i8*)
          catch i8* null
  %10 = extractvalue { i8*, i32 } %9, 0
  %11 = extractvalue { i8*, i32 } %9, 1
  sync within %syncreg, label %ehcleanup24

ehcleanup24:                                      ; preds = %lpad16, %lpad11
  %ehselector.slot.0 = phi i32 [ %11, %lpad16 ], [ %8, %lpad11 ]
  %exn.slot.0 = phi i8* [ %10, %lpad16 ], [ %7, %lpad11 ]
  sync within %syncreg, label %ehcleanup26
; CHECK: {{^ehcleanup24}}:
; CHECK: call void @__cilk_sync(%struct.__cilkrts_stack_frame* %[[CILKSF]])
; CHECK-NEXT: br label %ehcleanup26

ehcleanup26:                                      ; preds = %ehcleanup24, %lpad
  %ehselector.slot.1 = phi i32 [ %ehselector.slot.0, %ehcleanup24 ], [ %4, %lpad ]
  %exn.slot.1 = phi i8* [ %exn.slot.0, %ehcleanup24 ], [ %3, %lpad ]
  sync within %syncreg, label %catch.dispatch
; CHECK: {{^ehcleanup26}}:
; CHECK: call void @__cilk_sync(%struct.__cilkrts_stack_frame* %[[CILKSF]])
; CHECK-NEXT: br label %catch.dispatch

catch.dispatch:                                   ; preds = %ehcleanup26
  %12 = tail call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #11
  %matches = icmp eq i32 %ehselector.slot.1, %12
  %13 = tail call i8* @__cxa_begin_catch(i8* %exn.slot.1) #11
  br i1 %matches, label %catch34, label %catch

catch34:                                          ; preds = %catch.dispatch
  %14 = bitcast i8* %13 to i32*
  %15 = load i32, i32* %14, align 4, !tbaa !2
  invoke void @_Z10handle_exni(i32 %15)
          to label %invoke.cont37 unwind label %lpad36

invoke.cont37:                                    ; preds = %catch34
  tail call void @__cxa_end_catch() #11
  br label %try.cont

try.cont:                                         ; preds = %invoke.cont21, %invoke.cont37, %invoke.cont31
  ret void

catch:                                            ; preds = %catch.dispatch
  invoke void @_Z10handle_exni(i32 -1)
          to label %invoke.cont31 unwind label %lpad30

invoke.cont31:                                    ; preds = %catch
  tail call void @__cxa_end_catch()
  br label %try.cont

lpad30:                                           ; preds = %catch
  %16 = landingpad { i8*, i32 }
          cleanup
  %17 = extractvalue { i8*, i32 } %16, 0
  %18 = extractvalue { i8*, i32 } %16, 1
  invoke void @__cxa_end_catch()
          to label %eh.resume unwind label %terminate.lpad

lpad36:                                           ; preds = %catch34
  %19 = landingpad { i8*, i32 }
          cleanup
  %20 = extractvalue { i8*, i32 } %19, 0
  %21 = extractvalue { i8*, i32 } %19, 1
  tail call void @__cxa_end_catch() #11
  br label %eh.resume

eh.resume:                                        ; preds = %lpad30, %lpad36
  %ehselector.slot.2 = phi i32 [ %21, %lpad36 ], [ %18, %lpad30 ]
  %exn.slot.2 = phi i8* [ %20, %lpad36 ], [ %17, %lpad30 ]
  %lpad.val43 = insertvalue { i8*, i32 } undef, i8* %exn.slot.2, 0
  %lpad.val44 = insertvalue { i8*, i32 } %lpad.val43, i32 %ehselector.slot.2, 1
  resume { i8*, i32 } %lpad.val44
; CHECK: {{^eh.resume}}:
; CHECK: call void @__cilk_parent_epilogue(%struct.__cilkrts_stack_frame* %[[CILKSF]])
; CHECK-NEXT: resume { i8*, i32 }

terminate.lpad:                                   ; preds = %lpad30
  %22 = landingpad { i8*, i32 }
          catch i8* null
  %23 = extractvalue { i8*, i32 } %22, 0
  tail call void @__clang_call_terminate(i8* %23) #13
  unreachable
}


; CHECK-DAG: define private fastcc void @_Z14spawn_tryblocki.outline_det.achd13.otd1(i32
; CHECK: %[[ARG:[a-zA-Z0-9._]+]])
; CHECK: %[[CILKSF:.+]] = alloca %struct.__cilkrts_stack_frame
; CHECK: call void @__cilkrts_enter_frame_fast_1(%struct.__cilkrts_stack_frame* %[[CILKSF]])
; CHECK-NEXT: call void @__cilkrts_detach(%struct.__cilkrts_stack_frame* %[[CILKSF]])
; CHECK-NEXT: br label %[[BODY:.+]]

; CHECK: [[BODY]]:
; CHECK-NEXT: call i32 @_Z4quuzi(i32 {{.*}}%[[ARG]])
; CHECK-NEXT: br label %{{.+}}


; CHECK-DAG: define private fastcc void @_Z14spawn_tryblocki.outline_det.achd4.otd1(%class.Foo*
; CHECK: %[[ARG:[a-zA-Z0-9._]+]])
; CHECK: %[[CILKSF:.+]] = alloca %struct.__cilkrts_stack_frame
; CHECK: call void @__cilkrts_enter_frame_fast_1(%struct.__cilkrts_stack_frame* %[[CILKSF]])
; CHECK-NEXT: call void @__cilkrts_detach(%struct.__cilkrts_stack_frame* %[[CILKSF]])
; CHECK-NEXT: br label %[[BODY:.+]]

; CHECK: [[BODY]]:
; CHECK-NEXT: invoke i32 @_Z3barP3Foo(%class.Foo* {{.+}}%[[ARG]])
; CHECK-NEXT: to label %[[INVOKECONT:.+]] unwind label %[[LPAD:.+]]

; CHECK: [[LPAD]]:
; CHECK-NEXT: %[[LPADVAL:.+]] = landingpad { i8*, i32 }
; CHECK-NEXT: catch i8* null
; CHECK: %[[EXNDATA:.+]] = extractvalue { i8*, i32 } %[[LPADVAL]], 0
; CHECK-NEXT: %[[FLAGSADDR:.+]] = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %[[CILKSF]], i32 0, i32 0
; CHECK-NEXT: %[[FLAGS:.+]] = load {{.+}} i32, i32* %[[FLAGSADDR]]
; CHECK-NEXT: %[[NEWFLAGS:.+]] = or i32 %[[FLAGS]],
; CHECK: store {{.+}} i32 %[[NEWFLAGS]],
; CHECK: %[[EXNDATAADDR:.+]] = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %[[CILKSF]], i32 0, i32 4
; CHECK-NEXT: store {{.+}} i8* %[[EXNDATA]], i8** %[[EXNDATAADDR]]
; CHECK-NEXT: call void @__cilk_parent_epilogue(%struct.__cilkrts_stack_frame* %[[CILKSF]])
; CHECK-NEXT: resume { i8*, i32 } %[[LPADVAL]]
 

; CHECK-DAG: define private fastcc void @_Z18spawn_throw_inlinei.outline_det.achd.otd1(%class.Foo*
; CHECK: %[[ARG:[a-zA-Z0-9._]+]])
; CHECK: %[[CILKSF:.+]] = alloca %struct.__cilkrts_stack_frame
; CHECK: call void @__cilkrts_enter_frame_fast_1(%struct.__cilkrts_stack_frame* %[[CILKSF]])
; CHECK-NEXT: call void @__cilkrts_detach(%struct.__cilkrts_stack_frame* %[[CILKSF]])
; CHECK-NEXT: br label %[[BODY:.+]]

; CHECK: [[BODY]]:
; CHECK-NEXT: invoke i32 @_Z3barP3Foo(%class.Foo* {{.+}}%[[ARG]])
; CHECK-NEXT: to label %[[INVOKECONT:.+]] unwind label %[[LPAD:.+]]

; CHECK: [[LPAD]]:
; CHECK-NEXT: landingpad { i8*, i32 }
; CHECK-NEXT: catch i8* bitcast (i8** @_ZTIi to i8*)
; CHECK-NEXT: catch i8* null
; CHECK: br i1 %{{.+}}, label %[[CATCHIN:.+]], label %[[RESUMEIN:.+]]

; CHECK: [[RESUMEIN]]:
; CHECK: %[[EXNDATA:.+]] = extractvalue { i8*, i32 } %[[LPADVAL:.+]], 0
; CHECK-NEXT: %[[FLAGSADDR:.+]] = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %[[CILKSF]], i32 0, i32 0
; CHECK-NEXT: %[[FLAGS:.+]] = load {{.+}} i32, i32* %[[FLAGSADDR]]
; CHECK-NEXT: %[[NEWFLAGS:.+]] = or i32 %[[FLAGS]], 16
; CHECK-NEXT: %[[FLAGSADDR2:.+]] = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %[[CILKSF]], i32 0, i32 0
; CHECK-NEXT: store {{.+}} i32 %[[NEWFLAGS]], i32* %[[FLAGSADDR2]]
; CHECK-NEXT: %[[EXNDATAADDR:.+]] = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %[[CILKSF]], i32 0, i32 4
; CHECK-NEXT: store {{.+}} i8* %[[EXNDATA]], i8** %[[EXNDATAADDR]]
; CHECK-NEXT: call void @__cilk_parent_epilogue(%struct.__cilkrts_stack_frame* %[[CILKSF]])
; CHECK-NEXT: resume { i8*, i32 } %[[LPADVAL]]

; CHECK: [[CATCHIN]]:
; CHECK: invoke void @_Z10handle_exni(
; CHECK-NEXT: to label %[[INVOKECONT:.+]] unwind label %[[LPADIN:.+]]

; CHECK: [[LPADIN]]:
; CHECK: br label %[[RESUMEIN]]

; CHECK-DAG: define private fastcc void @_Z12spawn_excepti.outline_det.achd.otd1(%class.Foo*
; CHECK: %[[ARG:[a-zA-Z0-9._]+]])
; CHECK: %[[CILKSF:.+]] = alloca %struct.__cilkrts_stack_frame
; CHECK: call void @__cilkrts_enter_frame_fast_1(%struct.__cilkrts_stack_frame* %[[CILKSF]])
; CHECK-NEXT: call void @__cilkrts_detach(%struct.__cilkrts_stack_frame* %[[CILKSF]])
; CHECK-NEXT: br label %[[BODY:.+]]

; CHECK: [[BODY]]:
; CHECK-NEXT: invoke i32 @_Z3barP3Foo(%class.Foo* {{.+}}%[[ARG]])
; CHECK-NEXT: to label %[[INVOKECONT:.+]] unwind label %[[LPAD:.+]]

; CHECK: [[LPAD]]:
; CHECK-NEXT: %[[LPADVAL:.+]] = landingpad { i8*, i32 }
; CHECK-NEXT: catch i8* null
; CHECK: %[[EXNDATA:.+]] = extractvalue { i8*, i32 } %[[LPADVAL]], 0
; CHECK-NEXT: %[[FLAGSADDR:.+]] = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %[[CILKSF]], i32 0, i32 0
; CHECK-NEXT: %[[FLAGS:.+]] = load {{.+}} i32, i32* %[[FLAGSADDR]]
; CHECK-NEXT: %[[NEWFLAGS:.+]] = or i32 %[[FLAGS]], 16
; CHECK-NEXT: %[[FLAGSADDR2:.+]] = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %[[CILKSF]], i32 0, i32 0
; CHECK-NEXT: store {{.+}} i32 %[[NEWFLAGS]], i32* %[[FLAGSADDR2]]
; CHECK-NEXT: %[[EXNDATAADDR:.+]] = getelementptr inbounds %struct.__cilkrts_stack_frame, %struct.__cilkrts_stack_frame* %[[CILKSF]], i32 0, i32 4
; CHECK-NEXT: store {{.+}} i8* %[[EXNDATA]], i8** %[[EXNDATAADDR]]
; CHECK-NEXT: call void @__cilk_parent_epilogue(%struct.__cilkrts_stack_frame* %[[CILKSF]])
; CHECK-NEXT: resume { i8*, i32 } %[[LPADVAL]]

attributes #0 = { alwaysinline uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nobuiltin "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { noinline noreturn nounwind }
attributes #9 = { argmemonly nounwind }
attributes #10 = { argmemonly }
attributes #11 = { nounwind }
attributes #12 = { builtin }
attributes #13 = { noreturn nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 6.0.0 (git@github.com:wsmoses/Tapir-Clang.git d5d865dfb510d91f47fc5257febec4f52eb1afcb) (git@github.com:wsmoses/Tapir-LLVM.git 3284409d84c07572b3bb2b6f1ef6e73dd0152fc6)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = distinct !{!6, !7}
!7 = !{!"tapir.loop.spawn.strategy", i32 1}
!8 = distinct !{!8, !7}
!9 = distinct !{!9, !7}
!10 = distinct !{!10, !7}
!11 = distinct !{!11, !7}
