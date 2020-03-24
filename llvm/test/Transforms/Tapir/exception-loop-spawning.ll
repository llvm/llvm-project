; RUN: opt < %s -loop-spawning-ti -S | FileCheck %s
; RUN: opt < %s -passes=loop-spawning -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

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
  %3 = tail call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #10
  %matches = icmp eq i32 %2, %3
  br i1 %matches, label %catch, label %eh.resume

catch:                                            ; preds = %lpad
  %4 = tail call i8* @__cxa_begin_catch(i8* %1) #10
  %5 = bitcast i8* %4 to i32*
  %6 = load i32, i32* %5, align 4, !tbaa !2
  invoke void @_Z10handle_exni(i32 %6)
          to label %invoke.cont2 unwind label %lpad1

invoke.cont2:                                     ; preds = %catch
  tail call void @__cxa_end_catch() #10
  br label %try.cont

try.cont:                                         ; preds = %invoke.cont2, %entry
  ret i32 0

lpad1:                                            ; preds = %catch
  %7 = landingpad { i8*, i32 }
          cleanup
  %8 = extractvalue { i8*, i32 } %7, 0
  %9 = extractvalue { i8*, i32 } %7, 1
  tail call void @__cxa_end_catch() #10
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
declare i32 @_Z4quuzi(i32) local_unnamed_addr #3

; Function Attrs: nobuiltin
declare noalias nonnull i8* @_Znwm(i64) local_unnamed_addr #4

; Function Attrs: norecurse nounwind uwtable
define linkonce_odr void @_ZN3FooC2Ev(%class.Foo* %this) unnamed_addr #5 comdat align 2 {
entry:
  ret void
}

; Function Attrs: noinline noreturn nounwind
define linkonce_odr hidden void @__clang_call_terminate(i8*) local_unnamed_addr #6 comdat {
  %2 = tail call i8* @__cxa_begin_catch(i8* %0) #10
  tail call void @_ZSt9terminatev() #11
  unreachable
}

declare void @_ZSt9terminatev() local_unnamed_addr

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #7

; Function Attrs: uwtable
define void @_Z18parallelfor_excepti(i32 %n) local_unnamed_addr #8 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-LABEL: define void @_Z18parallelfor_excepti(i32 %n)
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp30 = icmp sgt i32 %n, 0
  br i1 %cmp30, label %pfor.detach.preheader, label %pfor.cond.cleanup

pfor.detach.preheader:                            ; preds = %entry
  br label %pfor.detach
; CHECK: invoke fastcc void @_Z18parallelfor_excepti.outline_pfor.detach.ls1(
; CHECK-NEXT: to label %{{.+}} unwind label %lpad7.loopexit

pfor.cond.cleanup.loopexit:                       ; preds = %pfor.inc
  br label %pfor.cond.cleanup

pfor.cond.cleanup:                                ; preds = %pfor.cond.cleanup.loopexit, %entry
  sync within %syncreg, label %sync.continue

pfor.detach:                                      ; preds = %pfor.inc, %pfor.detach.preheader
  %__begin.031 = phi i32 [ %inc, %pfor.inc ], [ 0, %pfor.detach.preheader ]
  detach within %syncreg, label %pfor.body, label %pfor.inc unwind label %lpad7.loopexit

pfor.body:                                        ; preds = %pfor.detach
  %call = invoke i8* @_Znwm(i64 1) #12
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %pfor.body
  %0 = bitcast i8* %call to %class.Foo*
  tail call void @_ZN3FooC2Ev(%class.Foo* nonnull %0)
  %call6 = invoke i32 @_Z3barP3Foo(%class.Foo* nonnull %0)
          to label %pfor.preattach unwind label %lpad

pfor.preattach:                                   ; preds = %invoke.cont
  reattach within %syncreg, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.preattach, %pfor.detach
  %inc = add nuw nsw i32 %__begin.031, 1
  %exitcond = icmp ne i32 %inc, %n
  br i1 %exitcond, label %pfor.detach, label %pfor.cond.cleanup.loopexit, !llvm.loop !6

lpad:                                             ; preds = %invoke.cont, %pfor.body
  %1 = landingpad { i8*, i32 }
          catch i8* null
  invoke void @llvm.detached.rethrow.sl_p0i8i32s(token %syncreg, { i8*, i32 } %1)
          to label %det.rethrow.unreachable unwind label %lpad7.loopexit.split-lp

det.rethrow.unreachable:                          ; preds = %lpad
  unreachable

lpad7.loopexit:                                   ; preds = %pfor.detach
  %lpad.loopexit = landingpad { i8*, i32 }
          cleanup
  br label %lpad7

lpad7.loopexit.split-lp:                          ; preds = %lpad
  %lpad.loopexit.split-lp = landingpad { i8*, i32 }
          cleanup
  br label %lpad7

lpad7:                                            ; preds = %lpad7.loopexit.split-lp, %lpad7.loopexit
  %lpad.phi = phi { i8*, i32 } [ %lpad.loopexit, %lpad7.loopexit ], [ %lpad.loopexit.split-lp, %lpad7.loopexit.split-lp ]
  sync within %syncreg, label %sync.continue12

sync.continue:                                    ; preds = %pfor.cond.cleanup
  ret void

sync.continue12:                                  ; preds = %lpad7
  resume { i8*, i32 } %lpad.phi
}

; Function Attrs: argmemonly
declare void @llvm.detached.rethrow.sl_p0i8i32s(token, { i8*, i32 }) #9

; Function Attrs: uwtable
define void @_Z20parallelfor_tryblocki(i32 %n) local_unnamed_addr #8 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-LABEL: define void @_Z20parallelfor_tryblocki(i32 %n)
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %syncreg3 = tail call token @llvm.syncregion.start()
  %cmp77 = icmp sgt i32 %n, 0
  br i1 %cmp77, label %pfor.detach.preheader, label %pfor.cond.cleanup

pfor.detach.preheader:                            ; preds = %entry
  br label %pfor.detach
; CHECK: call fastcc void @_Z20parallelfor_tryblocki.outline_pfor.detach.ls1(

pfor.cond.cleanup.loopexit:                       ; preds = %pfor.inc
  br label %pfor.cond.cleanup

pfor.cond.cleanup:                                ; preds = %pfor.cond.cleanup.loopexit, %entry
  sync within %syncreg, label %sync.continue

pfor.detach:                                      ; preds = %pfor.inc, %pfor.detach.preheader
  %__begin.078 = phi i32 [ %inc, %pfor.inc ], [ 0, %pfor.detach.preheader ]
  detach within %syncreg, label %pfor.body, label %pfor.inc

pfor.body:                                        ; preds = %pfor.detach
  %call = tail call i32 @_Z4quuzi(i32 %__begin.078) #10
  reattach within %syncreg, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.body, %pfor.detach
  %inc = add nuw nsw i32 %__begin.078, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %pfor.cond.cleanup.loopexit, label %pfor.detach, !llvm.loop !8

sync.continue:                                    ; preds = %pfor.cond.cleanup
  %cmp1275 = icmp sgt i32 %n, 0
  br i1 %cmp1275, label %pfor.detach14.preheader, label %pfor.cond.cleanup13

pfor.detach14.preheader:                          ; preds = %sync.continue
  br label %pfor.detach14
; CHECK: invoke fastcc void @_Z20parallelfor_tryblocki.outline_pfor.detach14.ls1(
; CHECK-NEXT: to label %{{.+}} unwind label %lpad28.loopexit

pfor.cond.cleanup13.loopexit:                     ; preds = %pfor.inc26
  br label %pfor.cond.cleanup13

pfor.cond.cleanup13:                              ; preds = %pfor.cond.cleanup13.loopexit, %sync.continue
  sync within %syncreg3, label %try.cont

pfor.detach14:                                    ; preds = %pfor.inc26, %pfor.detach14.preheader
  %__begin5.076 = phi i32 [ %inc27, %pfor.inc26 ], [ 0, %pfor.detach14.preheader ]
  detach within %syncreg3, label %pfor.body19, label %pfor.inc26 unwind label %lpad28.loopexit

pfor.body19:                                      ; preds = %pfor.detach14
  %call20 = invoke i8* @_Znwm(i64 1) #12
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %pfor.body19
  %0 = bitcast i8* %call20 to %class.Foo*
  tail call void @_ZN3FooC2Ev(%class.Foo* nonnull %0)
  %call24 = invoke i32 @_Z3barP3Foo(%class.Foo* nonnull %0)
          to label %pfor.preattach25 unwind label %lpad

pfor.preattach25:                                 ; preds = %invoke.cont
  reattach within %syncreg3, label %pfor.inc26

pfor.inc26:                                       ; preds = %pfor.preattach25, %pfor.detach14
  %inc27 = add nuw nsw i32 %__begin5.076, 1
  %exitcond1 = icmp ne i32 %inc27, %n
  br i1 %exitcond1, label %pfor.detach14, label %pfor.cond.cleanup13.loopexit, !llvm.loop !9

lpad:                                             ; preds = %invoke.cont, %pfor.body19
  %1 = landingpad { i8*, i32 }
          catch i8* null
  invoke void @llvm.detached.rethrow.sl_p0i8i32s(token %syncreg3, { i8*, i32 } %1)
          to label %det.rethrow.unreachable unwind label %lpad28.loopexit.split-lp

det.rethrow.unreachable:                          ; preds = %lpad
  unreachable

lpad28.loopexit:                                  ; preds = %pfor.detach14
  %lpad.loopexit = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIi to i8*)
          catch i8* null
  br label %lpad28

lpad28.loopexit.split-lp:                         ; preds = %lpad
  %lpad.loopexit.split-lp = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIi to i8*)
          catch i8* null
  br label %lpad28

lpad28:                                           ; preds = %lpad28.loopexit.split-lp, %lpad28.loopexit
  %lpad.phi = phi { i8*, i32 } [ %lpad.loopexit, %lpad28.loopexit ], [ %lpad.loopexit.split-lp, %lpad28.loopexit.split-lp ]
  %2 = extractvalue { i8*, i32 } %lpad.phi, 0
  %3 = extractvalue { i8*, i32 } %lpad.phi, 1
  sync within %syncreg3, label %sync.continue34

sync.continue34:                                  ; preds = %lpad28
  %4 = tail call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #10
  %matches = icmp eq i32 %3, %4
  %5 = tail call i8* @__cxa_begin_catch(i8* %2) #10
  br i1 %matches, label %catch45, label %catch

catch45:                                          ; preds = %sync.continue34
  %6 = bitcast i8* %5 to i32*
  %7 = load i32, i32* %6, align 4, !tbaa !2
  invoke void @_Z10handle_exni(i32 %7)
          to label %invoke.cont48 unwind label %lpad47

invoke.cont48:                                    ; preds = %catch45
  tail call void @__cxa_end_catch() #10
  br label %try.cont

try.cont:                                         ; preds = %invoke.cont42, %invoke.cont48, %pfor.cond.cleanup13
  ret void

catch:                                            ; preds = %sync.continue34
  invoke void @_Z10handle_exni(i32 -1)
          to label %invoke.cont42 unwind label %lpad41

invoke.cont42:                                    ; preds = %catch
  tail call void @__cxa_end_catch()
  br label %try.cont

lpad41:                                           ; preds = %catch
  %8 = landingpad { i8*, i32 }
          cleanup
  %9 = extractvalue { i8*, i32 } %8, 0
  %10 = extractvalue { i8*, i32 } %8, 1
  invoke void @__cxa_end_catch()
          to label %eh.resume unwind label %terminate.lpad

lpad47:                                           ; preds = %catch45
  %11 = landingpad { i8*, i32 }
          cleanup
  %12 = extractvalue { i8*, i32 } %11, 0
  %13 = extractvalue { i8*, i32 } %11, 1
  tail call void @__cxa_end_catch() #10
  br label %eh.resume

eh.resume:                                        ; preds = %lpad47, %lpad41
  %ehselector.slot30.0 = phi i32 [ %13, %lpad47 ], [ %10, %lpad41 ]
  %exn.slot29.0 = phi i8* [ %12, %lpad47 ], [ %9, %lpad41 ]
  %lpad.val53 = insertvalue { i8*, i32 } undef, i8* %exn.slot29.0, 0
  %lpad.val54 = insertvalue { i8*, i32 } %lpad.val53, i32 %ehselector.slot30.0, 1
  resume { i8*, i32 } %lpad.val54

terminate.lpad:                                   ; preds = %lpad41
  %14 = landingpad { i8*, i32 }
          catch i8* null
  %15 = extractvalue { i8*, i32 } %14, 0
  tail call void @__clang_call_terminate(i8* %15) #11
  unreachable
}

; Function Attrs: uwtable
define void @_Z27parallelfor_tryblock_inlinei(i32 %n) local_unnamed_addr #8 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-LABEL: @_Z27parallelfor_tryblock_inlinei(i32 %n)
entry:
  %syncreg = tail call token @llvm.syncregion.start()
  %cmp48 = icmp sgt i32 %n, 0
  br i1 %cmp48, label %pfor.detach.preheader, label %pfor.cond.cleanup

pfor.detach.preheader:                            ; preds = %entry
  br label %pfor.detach
; CHECK: invoke fastcc void @_Z27parallelfor_tryblock_inlinei.outline_pfor.detach.ls1(
; CHECK-NEXT: to label %{{.+}} unwind label %lpad7.loopexit

pfor.cond.cleanup.loopexit:                       ; preds = %pfor.inc
  br label %pfor.cond.cleanup

pfor.cond.cleanup:                                ; preds = %pfor.cond.cleanup.loopexit, %entry
  sync within %syncreg, label %try.cont

pfor.detach:                                      ; preds = %pfor.inc, %pfor.detach.preheader
  %__begin.049 = phi i32 [ %inc, %pfor.inc ], [ 0, %pfor.detach.preheader ]
  detach within %syncreg, label %pfor.body, label %pfor.inc unwind label %lpad7.loopexit

pfor.body:                                        ; preds = %pfor.detach
  %call = invoke i8* @_Znwm(i64 1) #12
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %pfor.body
  %0 = bitcast i8* %call to %class.Foo*
  tail call void @_ZN3FooC2Ev(%class.Foo* nonnull %0)
  %call.i = invoke i32 @_Z3barP3Foo(%class.Foo* nonnull %0)
          to label %pfor.preattach unwind label %lpad.i

lpad.i:                                           ; preds = %invoke.cont
  %1 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIi to i8*)
          catch i8* null
  %2 = extractvalue { i8*, i32 } %1, 0
  %3 = extractvalue { i8*, i32 } %1, 1
  %4 = tail call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #10
  %matches.i = icmp eq i32 %3, %4
  br i1 %matches.i, label %catch.i, label %eh.resume.i.loopexit

catch.i:                                          ; preds = %lpad.i
  %5 = tail call i8* @__cxa_begin_catch(i8* %2) #10
  %6 = bitcast i8* %5 to i32*
  %7 = load i32, i32* %6, align 4, !tbaa !2
  invoke void @_Z10handle_exni(i32 %7)
          to label %invoke.cont2.i unwind label %lpad1.i

invoke.cont2.i:                                   ; preds = %catch.i
  tail call void @__cxa_end_catch() #10
  br label %pfor.preattach

lpad1.i:                                          ; preds = %catch.i
  %8 = landingpad { i8*, i32 }
          catch i8* null
  %9 = extractvalue { i8*, i32 } %8, 0
  %10 = extractvalue { i8*, i32 } %8, 1
  tail call void @__cxa_end_catch() #10
  br label %eh.resume.i

eh.resume.i.loopexit:                             ; preds = %lpad.i
  %.lcssa2 = phi i8* [ %2, %lpad.i ]
  %.lcssa = phi i32 [ %3, %lpad.i ]
  br label %eh.resume.i

eh.resume.i:                                      ; preds = %eh.resume.i.loopexit, %lpad1.i
  %ehselector.slot.0.i = phi i32 [ %10, %lpad1.i ], [ %.lcssa, %eh.resume.i.loopexit ]
  %exn.slot.0.i = phi i8* [ %9, %lpad1.i ], [ %.lcssa2, %eh.resume.i.loopexit ]
  %lpad.val.i = insertvalue { i8*, i32 } undef, i8* %exn.slot.0.i, 0
  %lpad.val5.i = insertvalue { i8*, i32 } %lpad.val.i, i32 %ehselector.slot.0.i, 1
  br label %lpad.body

pfor.preattach:                                   ; preds = %invoke.cont2.i, %invoke.cont
  reattach within %syncreg, label %pfor.inc

pfor.inc:                                         ; preds = %pfor.preattach, %pfor.detach
  %inc = add nuw nsw i32 %__begin.049, 1
  %exitcond = icmp ne i32 %inc, %n
  br i1 %exitcond, label %pfor.detach, label %pfor.cond.cleanup.loopexit, !llvm.loop !10

lpad:                                             ; preds = %pfor.body
  %11 = landingpad { i8*, i32 }
          catch i8* null
  br label %lpad.body

lpad.body:                                        ; preds = %lpad, %eh.resume.i
  %eh.lpad-body = phi { i8*, i32 } [ %11, %lpad ], [ %lpad.val5.i, %eh.resume.i ]
  invoke void @llvm.detached.rethrow.sl_p0i8i32s(token %syncreg, { i8*, i32 } %eh.lpad-body)
          to label %det.rethrow.unreachable unwind label %lpad7.loopexit.split-lp

det.rethrow.unreachable:                          ; preds = %lpad.body
  unreachable

lpad7.loopexit:                                   ; preds = %pfor.detach
  %lpad.loopexit = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIi to i8*)
          catch i8* null
  br label %lpad7

lpad7.loopexit.split-lp:                          ; preds = %lpad.body
  %lpad.loopexit.split-lp = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIi to i8*)
          catch i8* null
  br label %lpad7

lpad7:                                            ; preds = %lpad7.loopexit.split-lp, %lpad7.loopexit
  %lpad.phi = phi { i8*, i32 } [ %lpad.loopexit, %lpad7.loopexit ], [ %lpad.loopexit.split-lp, %lpad7.loopexit.split-lp ]
  %12 = extractvalue { i8*, i32 } %lpad.phi, 0
  %13 = extractvalue { i8*, i32 } %lpad.phi, 1
  sync within %syncreg, label %sync.continue12

sync.continue12:                                  ; preds = %lpad7
  %14 = tail call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #10
  %matches = icmp eq i32 %13, %14
  %15 = tail call i8* @__cxa_begin_catch(i8* %12) #10
  br i1 %matches, label %catch22, label %catch

catch22:                                          ; preds = %sync.continue12
  %16 = bitcast i8* %15 to i32*
  %17 = load i32, i32* %16, align 4, !tbaa !2
  invoke void @_Z10handle_exni(i32 %17)
          to label %invoke.cont25 unwind label %lpad24

invoke.cont25:                                    ; preds = %catch22
  tail call void @__cxa_end_catch() #10
  br label %try.cont

try.cont:                                         ; preds = %invoke.cont19, %invoke.cont25, %pfor.cond.cleanup
  ret void

catch:                                            ; preds = %sync.continue12
  invoke void @_Z10handle_exni(i32 -1)
          to label %invoke.cont19 unwind label %lpad18

invoke.cont19:                                    ; preds = %catch
  tail call void @__cxa_end_catch()
  br label %try.cont

lpad18:                                           ; preds = %catch
  %18 = landingpad { i8*, i32 }
          cleanup
  %19 = extractvalue { i8*, i32 } %18, 0
  %20 = extractvalue { i8*, i32 } %18, 1
  invoke void @__cxa_end_catch()
          to label %eh.resume unwind label %terminate.lpad

lpad24:                                           ; preds = %catch22
  %21 = landingpad { i8*, i32 }
          cleanup
  %22 = extractvalue { i8*, i32 } %21, 0
  %23 = extractvalue { i8*, i32 } %21, 1
  tail call void @__cxa_end_catch() #10
  br label %eh.resume

eh.resume:                                        ; preds = %lpad24, %lpad18
  %ehselector.slot9.0 = phi i32 [ %23, %lpad24 ], [ %20, %lpad18 ]
  %exn.slot8.0 = phi i8* [ %22, %lpad24 ], [ %19, %lpad18 ]
  %lpad.val30 = insertvalue { i8*, i32 } undef, i8* %exn.slot8.0, 0
  %lpad.val31 = insertvalue { i8*, i32 } %lpad.val30, i32 %ehselector.slot9.0, 1
  resume { i8*, i32 } %lpad.val31

terminate.lpad:                                   ; preds = %lpad18
  %24 = landingpad { i8*, i32 }
          catch i8* null
  %25 = extractvalue { i8*, i32 } %24, 0
  tail call void @__clang_call_terminate(i8* %25) #11
  unreachable
}

; CHECK-LABEL: define private fastcc void @_Z18parallelfor_excepti.outline_pfor.detach.ls1(
; CHECK: %[[SYNCREG:.+]] = tail call token @llvm.syncregion.start()
; CHECK: detach within %[[SYNCREG]], label %[[RECURDET:.+]], label %[[RECURCONT:.+]] unwind label %[[RECURUW:.+]]

; CHECK: [[RECURDET]]:
; CHECK-NEXT: invoke fastcc void @_Z18parallelfor_excepti.outline_pfor.detach.ls1(
; CHECK-NEXT: to label %[[INVOKECONT:.+]] unwind label %[[TASKLPAD:.+]]

; CHECK: [[INVOKECONT]]:
; CHECK-NEXT: reattach within %[[SYNCREG]], label %[[RECURCONT]]

; CHECK: [[RECURUW]]:
; CHECK-NEXT: landingpad [[LPADTYPE:.+]]
; CHECK: sync within %[[SYNCREG]], label %[[SYNCCONT:.+]]

; CHECK: [[SYNCCONT]]:
; CHECK-NEXT: resume [[LPADTYPE]] %{{.+}}

; CHECK: {{^pfor.body.ls1}}:
; CHECK-NEXT: %[[NEWRET:.+]] = invoke i8* @_Znwm(i64 1)
; CHECK-NEXT: to label %invoke.cont.ls1 unwind label %lpad.ls1
; CHECK: {{^lpad.ls1}}:
; CHECK-NEXT: landingpad [[LPADTYPE]]
; CHECK-NEXT: catch {{.+}} null
; CHECK-NEXT: sync within %[[SYNCREG]], label %lpad.ls1.split
; CHECK: {{^lpad.ls1.split}}:
; CHECK-NEXT: resume [[LPADTYPE]]
; CHECK: {{^invoke.cont.ls1}}:
; CHECK-NEXT: %[[FOOARG:.+]] = bitcast i8* %[[NEWRET]] to %class.Foo*
; CHECK-NEXT: tail call void @_ZN3FooC2Ev(%class.Foo* nonnull %[[FOOARG]])
; CHECK-NEXT: %call6.ls1 = invoke i32 @_Z3barP3Foo(%class.Foo* nonnull %[[FOOARG]])
; CHECK-NEXT: to label %pfor.preattach.ls1 unwind label %lpad.ls1

; CHECK: [[TASKLPAD]]:
; CHECK-NEXT: landingpad [[LPADTYPE]]
; CHECK: invoke void @llvm.detached.rethrow
; CHECK: (token %[[SYNCREG]], [[LPADTYPE]] %{{.+}})


; CHECK-LABEL: define private fastcc void @_Z20parallelfor_tryblocki.outline_pfor.detach14.ls1(
; CHECK: %[[SYNCREG:.+]] = tail call token @llvm.syncregion.start()
; CHECK: detach within %[[SYNCREG]], label %[[RECURDET:.+]], label %[[RECURCONT:.+]] unwind label %[[RECURUW:.+]]
; CHECK: [[RECURDET]]:
; CHECK-NEXT: invoke fastcc void @_Z20parallelfor_tryblocki.outline_pfor.detach14.ls1(
; CHECK-NEXT: to label %[[INVOKECONT:.+]] unwind label %[[TASKLPAD:.+]]

; CHECK: [[INVOKECONT]]:
; CHECK-NEXT: reattach within %[[SYNCREG]], label %[[RECURCONT]]

; CHECK: [[RECURUW]]:
; CHECK-NEXT: landingpad [[LPADTYPE:.+]]
; CHECK: sync within %[[SYNCREG]], label %[[SYNCCONT:.+]]

; CHECK: [[SYNCCONT]]:
; CHECK-NEXT: resume [[LPADTYPE]] %{{.+}}

; CHECK: {{^pfor.body19.ls1}}:
; CHECK-NEXT: %[[NEWRET:.+]] = invoke i8* @_Znwm(i64 1)
; CHECK-NEXT: to label %invoke.cont.ls1 unwind label %lpad.ls1
; CHECK: {{^lpad.ls1}}:
; CHECK-NEXT: landingpad { i8*, i32 }
; CHECK-NEXT: catch {{.+}} null
; CHECK-NEXT: sync within %[[SYNCREG]], label %lpad.ls1.split
; CHECK: {{^lpad.ls1.split}}:
; CHECK-NEXT: resume [[LPADTYPE]] %{{.+}}
; CHECK: {{^invoke.cont.ls1}}:
; CHECK-NEXT: %[[FOOARG:.+]] = bitcast i8* %[[NEWRET]] to %class.Foo*
; CHECK-NEXT: tail call void @_ZN3FooC2Ev(%class.Foo* nonnull %[[FOOARG]])
; CHECK-NEXT: %call24.ls1 = invoke i32 @_Z3barP3Foo(%class.Foo* nonnull %[[FOOARG]])
; CHECK-NEXT: to label %pfor.preattach25.ls1 unwind label %lpad.ls1

; CHECK: [[TASKLPAD]]:
; CHECK-NEXT: landingpad [[LPADTYPE]]
; CHECK: invoke void @llvm.detached.rethrow
; CHECK: (token %[[SYNCREG]], [[LPADTYPE]] %{{.+}})


; CHECK-LABEL: define private fastcc void @_Z20parallelfor_tryblocki.outline_pfor.detach.ls1(
; CHECK-NOT: invoke
; CHECK-NOT: resume
; CHECK-NOT: detached.rethrow


; CHECK-LABEL: define private fastcc void @_Z27parallelfor_tryblock_inlinei.outline_pfor.detach.ls1(
; CHECK: %[[SYNCREG:.+]] = tail call token @llvm.syncregion.start()
; CHECK: detach within %[[SYNCREG]], label %[[RECURDET:.+]], label %[[RECURCONT:.+]] unwind label %[[RECURUW:.+]]
; CHECK: [[RECURDET]]:
; CHECK-NEXT: invoke fastcc void @_Z27parallelfor_tryblock_inlinei.outline_pfor.detach.ls1(
; CHECK-NEXT: to label %[[INVOKECONT:.+]] unwind label %[[TASKLPAD:.+]]

; CHECK: [[INVOKECONT]]:
; CHECK-NEXT: reattach within %[[SYNCREG]], label %[[RECURCONT]]

; CHECK: [[RECURUW]]:
; CHECK-NEXT: landingpad [[LPADTYPE:.+]]
; CHECK: sync within %[[SYNCREG]], label %[[SYNCCONT:.+]]

; CHECK: [[SYNCCONT]]:
; CHECK-NEXT: resume [[LPADTYPE]] %{{.+}}

; CHECK: {{^pfor.body.ls1}}:
; CHECK-NEXT: %[[NEWRET:.+]] = invoke i8* @_Znwm(i64 1)
; CHECK-NEXT: to label %invoke.cont.ls1 unwind label %lpad.ls1
; CHECK: {{^lpad.ls1}}:
; CHECK-NEXT: landingpad { i8*, i32 }
; CHECK-NEXT: catch {{.+}} null
; CHECK-NEXT: br label %lpad.body.ls1
; CHECK: {{^lpad.body.ls1}}:
; CHECK: sync within %[[SYNCREG]], label %[[RESUMEDST:.+]]
; CHECK: [[RESUMEDST]]:
; CHECK-NEXT: resume [[LPADTYPE]] %{{.+}}
; CHECK: {{^invoke.cont.ls1}}:
; CHECK-NEXT: %[[FOOARG:.+]] = bitcast i8* %[[NEWRET]] to %class.Foo*
; CHECK-NEXT: tail call void @_ZN3FooC2Ev(%class.Foo* nonnull %[[FOOARG]])
; CHECK-NEXT: %call.i.ls1 = invoke i32 @_Z3barP3Foo(%class.Foo* nonnull %[[FOOARG]])
; CHECK-NEXT: to label %pfor.preattach.ls1 unwind label %lpad.i.ls1
; CHECK: {{^lpad.i.ls1}}:
; CHECK-NEXT: landingpad { i8*, i32 }
; CHECK-NEXT: catch {{.+}} bitcast
; CHECK-NEXT: catch {{.+}} null
; CHECK: br i1 %{{.+}}, label %[[CATCHIN:.+]], label %[[RESUMEIN:.+]]
; CHECK: [[RESUMEIN]]:
; CHECK: br label %[[RESUMECOLLECT:.+]]
; CHECK: [[RESUMECOLLECT]]:
; CHECK: br label %lpad.body.ls1
; CHECK: [[CATCHIN]]:
; CHECK: tail call i8* @__cxa_begin_catch(
; CHECK: invoke void @_Z10handle_exni(
; CHECK: br label %[[RESUMECOLLECT]]

; CHECK: [[TASKLPAD]]:
; CHECK-NEXT: landingpad [[LPADTYPE]]
; CHECK: invoke void @llvm.detached.rethrow
; CHECK: (token %[[SYNCREG]], [[LPADTYPE]] %{{.+}})

attributes #0 = { alwaysinline uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nobuiltin "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { noinline noreturn nounwind }
attributes #7 = { argmemonly nounwind }
attributes #8 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { argmemonly }
attributes #10 = { nounwind }
attributes #11 = { noreturn nounwind }
attributes #12 = { builtin }

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
