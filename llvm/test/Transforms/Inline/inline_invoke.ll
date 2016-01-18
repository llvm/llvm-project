; RUN: opt < %s -inline -S | FileCheck %s

; Test that the inliner correctly handles inlining into invoke sites
; by appending selectors and forwarding _Unwind_Resume directly to the
; enclosing landing pad.

;; Test 0 - basic functionality.

%struct.A = type { i8 }

@_ZTIi = external constant i8*

declare void @_ZN1AC1Ev(%struct.A*)

declare void @_ZN1AD1Ev(%struct.A*)

declare void @use(i32) nounwind

declare void @opaque()

declare i32 @llvm.eh.typeid.for(i8*) nounwind

declare i32 @__gxx_personality_v0(...)

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()

declare void @_ZSt9terminatev()

define internal void @test0_in() alwaysinline uwtable ssp personality i32 (...)* @__gxx_personality_v0 {
entry:
  %a = alloca %struct.A, align 1
  %b = alloca %struct.A, align 1
  call void @_ZN1AC1Ev(%struct.A* %a)
  invoke void @_ZN1AC1Ev(%struct.A* %b)
          to label %invoke.cont unwind label %lpad

invoke.cont:
  invoke void @_ZN1AD1Ev(%struct.A* %b)
          to label %invoke.cont1 unwind label %lpad

invoke.cont1:
  call void @_ZN1AD1Ev(%struct.A* %a)
  ret void

lpad:
  %exn = landingpad {i8*, i32}
            cleanup
  invoke void @_ZN1AD1Ev(%struct.A* %a)
          to label %invoke.cont2 unwind label %terminate.lpad

invoke.cont2:
  resume { i8*, i32 } %exn

terminate.lpad:
  %exn1 = landingpad {i8*, i32}
            catch i8* null
  call void @_ZSt9terminatev() noreturn nounwind
  unreachable
}

define void @test0_out() uwtable ssp personality i32 (...)* @__gxx_personality_v0 {
entry:
  invoke void @test0_in()
          to label %ret unwind label %lpad

ret:
  ret void

lpad:                                             ; preds = %entry
  %exn = landingpad {i8*, i32}
            catch i8* bitcast (i8** @_ZTIi to i8*)
  %eh.exc = extractvalue { i8*, i32 } %exn, 0
  %eh.selector = extractvalue { i8*, i32 } %exn, 1
  %0 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) nounwind
  %1 = icmp eq i32 %eh.selector, %0
  br i1 %1, label %catch, label %eh.resume

catch:
  %ignored = call i8* @__cxa_begin_catch(i8* %eh.exc) nounwind
  call void @__cxa_end_catch() nounwind
  br label %ret

eh.resume:
  resume { i8*, i32 } %exn
}

; CHECK:    define void @test0_out()
; CHECK:      [[A:%.*]] = alloca %struct.A,
; CHECK:      [[B:%.*]] = alloca %struct.A,
; CHECK:      invoke void @_ZN1AC1Ev(%struct.A* [[A]])
; CHECK:      invoke void @_ZN1AC1Ev(%struct.A* [[B]])
; CHECK:      invoke void @_ZN1AD1Ev(%struct.A* [[B]])
; CHECK:      invoke void @_ZN1AD1Ev(%struct.A* [[A]])
; CHECK:      landingpad { i8*, i32 }
; CHECK-NEXT:    cleanup
; CHECK-NEXT:    catch i8* bitcast (i8** @_ZTIi to i8*)
; CHECK-NEXT: invoke void @_ZN1AD1Ev(%struct.A* [[A]])
; CHECK-NEXT:   to label %[[LBL:[^\s]+]] unwind
; CHECK: [[LBL]]:
; CHECK-NEXT: br label %[[LPAD:[^\s]+]]
; CHECK:      ret void
; CHECK:      landingpad { i8*, i32 }
; CHECK-NEXT:    catch i8* bitcast (i8** @_ZTIi to i8*)
; CHECK-NEXT: br label %[[LPAD]]
; CHECK: [[LPAD]]:
; CHECK-NEXT: phi { i8*, i32 } [
; CHECK-NEXT: extractvalue { i8*, i32 }
; CHECK-NEXT: extractvalue { i8*, i32 }
; CHECK-NEXT: call i32 @llvm.eh.typeid.for(


;; Test 1 - Correctly handle phis in outer landing pads.

define void @test1_out() uwtable ssp personality i32 (...)* @__gxx_personality_v0 {
entry:
  invoke void @test0_in()
          to label %cont unwind label %lpad

cont:
  invoke void @test0_in()
          to label %ret unwind label %lpad

ret:
  ret void

lpad:
  %x = phi i32 [ 0, %entry ], [ 1, %cont ]
  %y = phi i32 [ 1, %entry ], [ 4, %cont ]
  %exn = landingpad {i8*, i32}
            catch i8* bitcast (i8** @_ZTIi to i8*)
  %eh.exc = extractvalue { i8*, i32 } %exn, 0
  %eh.selector = extractvalue { i8*, i32 } %exn, 1
  %0 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) nounwind
  %1 = icmp eq i32 %eh.selector, %0
  br i1 %1, label %catch, label %eh.resume

catch:
  %ignored = call i8* @__cxa_begin_catch(i8* %eh.exc) nounwind
  call void @use(i32 %x)
  call void @use(i32 %y)
  call void @__cxa_end_catch() nounwind
  br label %ret

eh.resume:
  resume { i8*, i32 } %exn
}

; CHECK:    define void @test1_out()
; CHECK:      [[A2:%.*]] = alloca %struct.A,
; CHECK:      [[B2:%.*]] = alloca %struct.A,
; CHECK:      [[A1:%.*]] = alloca %struct.A,
; CHECK:      [[B1:%.*]] = alloca %struct.A,
; CHECK:      invoke void @_ZN1AC1Ev(%struct.A* [[A1]])
; CHECK-NEXT:   unwind label %[[LPAD:[^\s]+]]
; CHECK:      invoke void @_ZN1AC1Ev(%struct.A* [[B1]])
; CHECK-NEXT:   unwind label %[[LPAD1:[^\s]+]]
; CHECK:      invoke void @_ZN1AD1Ev(%struct.A* [[B1]])
; CHECK-NEXT:   unwind label %[[LPAD1]]
; CHECK:      invoke void @_ZN1AD1Ev(%struct.A* [[A1]])
; CHECK-NEXT:   unwind label %[[LPAD]]

; Inner landing pad from first inlining.
; CHECK:    [[LPAD1]]:
; CHECK-NEXT: [[LPADVAL1:%.*]] = landingpad { i8*, i32 }
; CHECK-NEXT:    cleanup
; CHECK-NEXT:    catch i8* bitcast (i8** @_ZTIi to i8*)
; CHECK-NEXT: invoke void @_ZN1AD1Ev(%struct.A* [[A1]])
; CHECK-NEXT:   to label %[[RESUME1:[^\s]+]] unwind
; CHECK: [[RESUME1]]:
; CHECK-NEXT: br label %[[LPAD_JOIN1:[^\s]+]]

; CHECK:      invoke void @_ZN1AC1Ev(%struct.A* [[A2]])
; CHECK-NEXT:   unwind label %[[LPAD]]
; CHECK:      invoke void @_ZN1AC1Ev(%struct.A* [[B2]])
; CHECK-NEXT:   unwind label %[[LPAD2:[^\s]+]]
; CHECK:      invoke void @_ZN1AD1Ev(%struct.A* [[B2]])
; CHECK-NEXT:   unwind label %[[LPAD2]]
; CHECK:      invoke void @_ZN1AD1Ev(%struct.A* [[A2]])
; CHECK-NEXT:   unwind label %[[LPAD]]

; Inner landing pad from second inlining.
; CHECK:    [[LPAD2]]:
; CHECK-NEXT: [[LPADVAL2:%.*]] = landingpad { i8*, i32 }
; CHECK-NEXT:   cleanup
; CHECK-NEXT:   catch i8* bitcast (i8** @_ZTIi to i8*)
; CHECK-NEXT: invoke void @_ZN1AD1Ev(%struct.A* [[A2]])
; CHECK-NEXT:   to label %[[RESUME2:[^\s]+]] unwind
; CHECK: [[RESUME2]]:
; CHECK-NEXT: br label %[[LPAD_JOIN2:[^\s]+]]

; CHECK:      ret void

; CHECK:    [[LPAD]]:
; CHECK-NEXT: [[X:%.*]] = phi i32 [ 0, %entry ], [ 0, {{%.*}} ], [ 1, %cont ], [ 1, {{%.*}} ]
; CHECK-NEXT: [[Y:%.*]] = phi i32 [ 1, %entry ], [ 1, {{%.*}} ], [ 4, %cont ], [ 4, {{%.*}} ]
; CHECK-NEXT: [[LPADVAL:%.*]] = landingpad { i8*, i32 }
; CHECK-NEXT:   catch i8* bitcast (i8** @_ZTIi to i8*)
; CHECK-NEXT: br label %[[LPAD_JOIN2]]

; CHECK: [[LPAD_JOIN2]]:
; CHECK-NEXT: [[XJ2:%.*]] = phi i32 [ [[X]], %[[LPAD]] ], [ 1, %[[RESUME2]] ]
; CHECK-NEXT: [[YJ2:%.*]] = phi i32 [ [[Y]], %[[LPAD]] ], [ 4, %[[RESUME2]] ]
; CHECK-NEXT: [[EXNJ2:%.*]] = phi { i8*, i32 } [ [[LPADVAL]], %[[LPAD]] ], [ [[LPADVAL2]], %[[RESUME2]] ]
; CHECK-NEXT: br label %[[LPAD_JOIN1]]

; CHECK: [[LPAD_JOIN1]]:
; CHECK-NEXT: [[XJ1:%.*]] = phi i32 [ [[XJ2]], %[[LPAD_JOIN2]] ], [ 0, %[[RESUME1]] ]
; CHECK-NEXT: [[YJ1:%.*]] = phi i32 [ [[YJ2]], %[[LPAD_JOIN2]] ], [ 1, %[[RESUME1]] ]
; CHECK-NEXT: [[EXNJ1:%.*]] = phi { i8*, i32 } [ [[EXNJ2]], %[[LPAD_JOIN2]] ], [ [[LPADVAL1]], %[[RESUME1]] ]
; CHECK-NEXT: extractvalue { i8*, i32 } [[EXNJ1]], 0
; CHECK-NEXT: [[SELJ1:%.*]] = extractvalue { i8*, i32 } [[EXNJ1]], 1
; CHECK-NEXT: [[T:%.*]] = call i32 @llvm.eh.typeid.for(
; CHECK-NEXT: icmp eq i32 [[SELJ1]], [[T]]

; CHECK:      call void @use(i32 [[XJ1]])
; CHECK:      call void @use(i32 [[YJ1]])

; CHECK:      resume { i8*, i32 }


;; Test 2 - Don't make invalid IR for inlines into landing pads without eh.exception calls
define void @test2_out() uwtable ssp personality i32 (...)* @__gxx_personality_v0 {
entry:
  invoke void @test0_in()
          to label %ret unwind label %lpad

ret:
  ret void

lpad:
  %exn = landingpad {i8*, i32}
            cleanup
  call void @_ZSt9terminatev()
  unreachable
}

; CHECK: define void @test2_out()
; CHECK:      [[A:%.*]] = alloca %struct.A,
; CHECK:      [[B:%.*]] = alloca %struct.A,
; CHECK:      invoke void @_ZN1AC1Ev(%struct.A* [[A]])
; CHECK-NEXT:   unwind label %[[LPAD:[^\s]+]]
; CHECK:      invoke void @_ZN1AC1Ev(%struct.A* [[B]])
; CHECK-NEXT:   unwind label %[[LPAD2:[^\s]+]]
; CHECK:      invoke void @_ZN1AD1Ev(%struct.A* [[B]])
; CHECK-NEXT:   unwind label %[[LPAD2]]
; CHECK:      invoke void @_ZN1AD1Ev(%struct.A* [[A]])
; CHECK-NEXT:   unwind label %[[LPAD]]


;; Test 3 - Deal correctly with split unwind edges.
define void @test3_out() uwtable ssp personality i32 (...)* @__gxx_personality_v0 {
entry:
  invoke void @test0_in()
          to label %ret unwind label %lpad

ret:
  ret void

lpad:
  %exn = landingpad {i8*, i32}
            catch i8* bitcast (i8** @_ZTIi to i8*)
  br label %lpad.cont

lpad.cont:
  call void @_ZSt9terminatev()
  unreachable
}

; CHECK: define void @test3_out()
; CHECK:      landingpad { i8*, i32 }
; CHECK-NEXT:    cleanup
; CHECK-NEXT:    catch i8* bitcast (i8** @_ZTIi to i8*)
; CHECK-NEXT: invoke void @_ZN1AD1Ev(
; CHECK-NEXT:   to label %[[L:[^\s]+]] unwind
; CHECK:    [[L]]:
; CHECK-NEXT: br label %[[JOIN:[^\s]+]]
; CHECK:    [[JOIN]]:
; CHECK-NEXT: phi { i8*, i32 }
; CHECK-NEXT: br label %lpad.cont
; CHECK:    lpad.cont:
; CHECK-NEXT: call void @_ZSt9terminatev()


;; Test 4 - Split unwind edges with a dominance problem
define void @test4_out() uwtable ssp personality i32 (...)* @__gxx_personality_v0 {
entry:
  invoke void @test0_in()
          to label %cont unwind label %lpad.crit

cont:
  invoke void @opaque()
          to label %ret unwind label %lpad

ret:
  ret void

lpad.crit:
  %exn = landingpad {i8*, i32}
            catch i8* bitcast (i8** @_ZTIi to i8*)
  call void @opaque() nounwind
  br label %terminate

lpad:
  %exn2 = landingpad {i8*, i32}
            catch i8* bitcast (i8** @_ZTIi to i8*)
  br label %terminate

terminate:
  %phi = phi i32 [ 0, %lpad.crit ], [ 1, %lpad ]
  call void @use(i32 %phi)
  call void @_ZSt9terminatev()
  unreachable
}

; CHECK: define void @test4_out()
; CHECK:      landingpad { i8*, i32 }
; CHECK-NEXT:    cleanup
; CHECK-NEXT:    catch i8* bitcast (i8** @_ZTIi to i8*)
; CHECK-NEXT: invoke void @_ZN1AD1Ev(
; CHECK-NEXT:   to label %[[L:[^\s]+]] unwind
; CHECK:    [[L]]:
; CHECK-NEXT: br label %[[JOIN:[^\s]+]]
; CHECK:      invoke void @opaque()
; CHECK-NEXT:                  unwind label %lpad
; CHECK:    lpad.crit:
; CHECK-NEXT: landingpad { i8*, i32 }
; CHECK-NEXT:   catch i8* bitcast (i8** @_ZTIi to i8*)
; CHECK-NEXT: br label %[[JOIN]]
; CHECK:    [[JOIN]]:
; CHECK-NEXT: phi { i8*, i32 }
; CHECK-NEXT: call void @opaque() [[NUW:#[0-9]+]]
; CHECK-NEXT: br label %[[FIX:[^\s]+]]
; CHECK:    lpad:
; CHECK-NEXT: landingpad { i8*, i32 }
; CHECK-NEXT:   catch i8* bitcast (i8** @_ZTIi to i8*)
; CHECK-NEXT: br label %[[FIX]]
; CHECK:    [[FIX]]:
; CHECK-NEXT: [[T1:%.*]] = phi i32 [ 0, %[[JOIN]] ], [ 1, %lpad ]
; CHECK-NEXT: call void @use(i32 [[T1]])
; CHECK-NEXT: call void @_ZSt9terminatev()

; CHECK: attributes [[NUW]] = { nounwind }
; CHECK: attributes #1 = { nounwind readnone }
; CHECK: attributes #2 = { ssp uwtable }
; CHECK: attributes #3 = { nounwind argmemonly }
; CHECK: attributes #4 = { noreturn nounwind }
