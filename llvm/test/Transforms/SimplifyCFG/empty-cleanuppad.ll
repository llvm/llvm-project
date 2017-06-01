; RUN: opt < %s -simplifycfg -S | FileCheck %s

; ModuleID = 'cppeh-simplify.cpp'
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc18.0.0"


; This case arises when two objects with empty destructors are cleaned up.
;
; void f1() { 
;   S a;
;   S b;
;   g(); 
; }
;
; In this case, both cleanup pads can be eliminated and the invoke can be
; converted to a call.
;
; CHECK: define void @f1()
; CHECK: entry:
; CHECK:   call void @g()
; CHECK:   ret void
; CHECK-NOT: cleanuppad
; CHECK: }
;
define void @f1() personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  invoke void @g() to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  ret void

ehcleanup:                                        ; preds = %entry
  %0 = cleanuppad within none []
  cleanupret from %0 unwind label %ehcleanup.1

ehcleanup.1:                                      ; preds = %ehcleanup
  %1 = cleanuppad within none []
  cleanupret from %1 unwind to caller
}


; This case arises when an object with an empty destructor must be cleaned up
; outside of a try-block and an object with a non-empty destructor must be
; cleaned up within the try-block.
;
; void f2() { 
;   S a;
;   try {
;     S2 b;
;     g();
;   } catch (...) {}
; }
;
; In this case, the outermost cleanup pad can be eliminated and the catch block
; should unwind to the caller (that is, exception handling continues with the
; parent frame of the caller).
;
; CHECK: define void @f2()
; CHECK: entry:
; CHECK:   invoke void @g()
; CHECK: ehcleanup:
; CHECK:   cleanuppad within none
; CHECK:   call void @"\01??1S2@@QEAA@XZ"(%struct.S2* %b)
; CHECK:   cleanupret from %0 unwind label %catch.dispatch
; CHECK: catch.dispatch:
; CHECK:   catchswitch within none [label %catch] unwind to caller
; CHECK: catch:
; CHECK:   catchpad
; CHECK:   catchret
; CHECK-NOT: cleanuppad
; CHECK: }
;
define void @f2() personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %b = alloca %struct.S2, align 1
  invoke void @g() to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  br label %try.cont

ehcleanup:                                        ; preds = %entry
  %0 = cleanuppad within none []
  call void @"\01??1S2@@QEAA@XZ"(%struct.S2* %b)
  cleanupret from %0 unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %ehcleanup
  %cs1 = catchswitch within none [label %catch] unwind label %ehcleanup.1

catch:                                            ; preds = %catch.dispatch
  %1 = catchpad within %cs1 [i8* null, i32 u0x40, i8* null]
  catchret from %1 to label %catchret.dest

catchret.dest:                                    ; preds = %catch
  br label %try.cont

try.cont:                                         ; preds = %catchret.dest, %invoke.cont
  ret void

ehcleanup.1:
  %2 = cleanuppad within none []
  cleanupret from %2 unwind to caller
}


; This case arises when an object with a non-empty destructor must be cleaned up
; outside of a try-block and an object with an empty destructor must be cleaned
; within the try-block.
;
; void f3() { 
;   S2 a;
;   try {
;     S b;
;     g();
;   } catch (...) {}
; }
;
; In this case the inner cleanup pad should be eliminated and the invoke of g()
; should unwind directly to the catchpad.
;
; CHECK-LABEL: define void @f3()
; CHECK: entry:
; CHECK:   invoke void @g()
; CHECK:           to label %try.cont unwind label %catch.dispatch
; CHECK: catch.dispatch:
; CHECK-NEXT: catchswitch within none [label %catch] unwind label %ehcleanup.1
; CHECK: catch:
; CHECK:   catchpad within %cs1 [i8* null, i32 64, i8* null]
; CHECK:   catchret
; CHECK: ehcleanup.1:
; CHECK:   cleanuppad
; CHECK:   call void @"\01??1S2@@QEAA@XZ"(%struct.S2* %a)
; CHECK:   cleanupret from %cp3 unwind to caller
; CHECK: }
;
define void @f3() personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %a = alloca %struct.S2, align 1
  invoke void @g() to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  br label %try.cont

ehcleanup:                                        ; preds = %entry
  %0 = cleanuppad within none []
  cleanupret from %0 unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %ehcleanup
  %cs1 = catchswitch within none [label %catch] unwind label %ehcleanup.1

catch:                                            ; preds = %catch.dispatch
  %cp2 = catchpad within %cs1 [i8* null, i32 u0x40, i8* null]
  catchret from %cp2 to label %catchret.dest

catchret.dest:                                    ; preds = %catch
  br label %try.cont

try.cont:                                         ; preds = %catchret.dest, %invoke.cont
  ret void

ehcleanup.1:
  %cp3 = cleanuppad within none []
  call void @"\01??1S2@@QEAA@XZ"(%struct.S2* %a)
  cleanupret from %cp3 unwind to caller
}


; This case arises when an object with an empty destructor may require cleanup
; from either inside or outside of a try-block.
;
; void f4() { 
;   S a;
;   g();
;   try {
;     g();
;   } catch (...) {}
; }
;
; In this case, the cleanuppad should be eliminated, the invoke outside of the
; catch block should be converted to a call (that is, that is, exception
; handling continues with the parent frame of the caller).)
;
; CHECK-LABEL: define void @f4()
; CHECK: entry:
; CHECK:   call void @g
; Note: The cleanuppad simplification will insert an unconditional branch here
;       but it will be eliminated, placing the following invoke in the entry BB. 
; CHECK:   invoke void @g()
; CHECK:           to label %try.cont unwind label %catch.dispatch
; CHECK: catch.dispatch:
; CHECK:   catchswitch within none [label %catch] unwind to caller
; CHECK: catch:
; CHECK:   catchpad
; CHECK:   catchret
; CHECK-NOT: cleanuppad
; CHECK: }
;
define void @f4() personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  invoke void @g()
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  invoke void @g()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %invoke.cont
  %cs1 = catchswitch within none [label %catch] unwind label %ehcleanup

catch:                                            ; preds = %catch.dispatch
  %0 = catchpad within %cs1 [i8* null, i32 u0x40, i8* null]
  catchret from %0 to label %try.cont

try.cont:                                         ; preds = %catch, %invoke.cont
  ret void

ehcleanup:
  %cp2 = cleanuppad within none []
  cleanupret from %cp2 unwind to caller
}

; This case tests simplification of an otherwise empty cleanup pad that contains
; a PHI node.
;
; int f6() {
;   int state = 1;
;   try {
;     S a;
;     g();
;     state = 2;
;     g();
;   } catch (...) {
;     return state;
;   }
;   return 0;
; }
;
; In this case, the cleanup pad should be eliminated and the PHI node in the
; cleanup pad should be sunk into the catch dispatch block.
;
; CHECK-LABEL: define i32 @f6()
; CHECK: entry:
; CHECK:   invoke void @g()
; CHECK: invoke.cont:
; CHECK:   invoke void @g()
; CHECK-NOT: ehcleanup:
; CHECK-NOT:   cleanuppad
; CHECK: catch.dispatch:
; CHECK:   %state.0 = phi i32 [ 2, %invoke.cont ], [ 1, %entry ]
; CHECK: }
define i32 @f6() personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  invoke void @g()
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  invoke void @g()
          to label %return unwind label %ehcleanup

ehcleanup:                                        ; preds = %invoke.cont, %entry
  %state.0 = phi i32 [ 2, %invoke.cont ], [ 1, %entry ]
  %0 = cleanuppad within none []
  cleanupret from %0 unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %ehcleanup
  %cs1 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %1 = catchpad within %cs1 [i8* null, i32 u0x40, i8* null]
  catchret from %1 to label %return

return:                                           ; preds = %invoke.cont, %catch
  %retval.0 = phi i32 [ %state.0, %catch ], [ 0, %invoke.cont ]
  ret i32 %retval.0
}

; This case tests another variation of simplification of an otherwise empty
; cleanup pad that contains a PHI node.
;
; int f7() {
;   int state = 1;
;   try {
;     g();
;     state = 2;
;     S a;
;     g();
;     state = 3;
;     g();
;   } catch (...) {
;     return state;
;   }
;   return 0;
; }
;
; In this case, the cleanup pad should be eliminated and the PHI node in the
; cleanup pad should be merged with the PHI node in the catch dispatch block.
;
; CHECK-LABEL: define i32 @f7()
; CHECK: entry:
; CHECK:   invoke void @g()
; CHECK: invoke.cont:
; CHECK:   invoke void @g()
; CHECK: invoke.cont.1:
; CHECK:   invoke void @g()
; CHECK-NOT: ehcleanup:
; CHECK-NOT:   cleanuppad
; CHECK: catch.dispatch:
; CHECK:   %state.1 = phi i32 [ 1, %entry ], [ 3, %invoke.cont.1 ], [ 2, %invoke.cont ]
; CHECK: }
define i32 @f7() personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  invoke void @g()
          to label %invoke.cont unwind label %catch.dispatch

invoke.cont:                                      ; preds = %entry
  invoke void @g()
          to label %invoke.cont.1 unwind label %ehcleanup

invoke.cont.1:                                    ; preds = %invoke.cont
  invoke void @g()
          to label %return unwind label %ehcleanup

ehcleanup:                                        ; preds = %invoke.cont.1, %invoke.cont
  %state.0 = phi i32 [ 3, %invoke.cont.1 ], [ 2, %invoke.cont ]
  %0 = cleanuppad within none []
  cleanupret from %0 unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %ehcleanup, %entry
  %state.1 = phi i32 [ %state.0, %ehcleanup ], [ 1, %entry ]
  %cs1 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %1 = catchpad within %cs1 [i8* null, i32 u0x40, i8* null]
  catchret from %1 to label %return

return:                                           ; preds = %invoke.cont.1, %catch
  %retval.0 = phi i32 [ %state.1, %catch ], [ 0, %invoke.cont.1 ]
  ret i32 %retval.0
}

; This case tests a scenario where an empty cleanup pad is not dominated by all
; of the predecessors of its successor, but the successor references a PHI node
; in the empty cleanup pad.
;
; Conceptually, the case being modeled is something like this:
;
; int f8() {
;   int x = 1;
;   try {
;     S a;
;     g();
;     x = 2;
; retry:
;     g();
;     return
;   } catch (...) {
;     use_x(x);
;   }
;   goto retry;
; }
;
; While that C++ syntax isn't legal, the IR below is.
;
; In this case, the PHI node that is sunk from ehcleanup to catch.dispatch
; should have an incoming value entry for path from 'foo' that references the
; PHI node itself.
;
; CHECK-LABEL: define void @f8()
; CHECK: entry:
; CHECK:   invoke void @g()
; CHECK: invoke.cont:
; CHECK:   invoke void @g()
; CHECK-NOT: ehcleanup:
; CHECK-NOT:   cleanuppad
; CHECK: catch.dispatch:
; CHECK:   %x = phi i32 [ 2, %invoke.cont ], [ 1, %entry ], [ %x, %catch.cont ] 
; CHECK: }
define void @f8() personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  invoke void @g()
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  invoke void @g()
          to label %return unwind label %ehcleanup

ehcleanup:                                        ; preds = %invoke.cont, %entry
  %x = phi i32 [ 2, %invoke.cont ], [ 1, %entry ]
  %0 = cleanuppad within none []
  cleanupret from %0 unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %ehcleanup, %catch.cont
  %cs1 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %1 = catchpad within %cs1 [i8* null, i32 u0x40, i8* null]
  call void @use_x(i32 %x)
  catchret from %1 to label %catch.cont

catch.cont:                                       ; preds = %catch
  invoke void @g()
          to label %return unwind label %catch.dispatch

return:                                           ; preds = %invoke.cont, %catch.cont
  ret void
}
; CHECK-LABEL: define i32 @f9()
; CHECK: entry:
; CHECK:   invoke void @"\01??1S2@@QEAA@XZ"(
; CHECK-NOT:   cleanuppad
; CHECK: catch.dispatch:
; CHECK: }
define i32 @f9() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  %s = alloca i8, align 1
  call void @llvm.lifetime.start(i64 1, i8* nonnull %s)
  %bc = bitcast i8* %s to %struct.S2*
  invoke void @"\01??1S2@@QEAA@XZ"(%struct.S2* %bc)
          to label %try.cont unwind label %ehcleanup

ehcleanup:
  %cleanup.pad = cleanuppad within none []
  call void @llvm.lifetime.end(i64 1, i8* nonnull %s)
  cleanupret from %cleanup.pad unwind label %catch.dispatch

catch.dispatch:
  %catch.switch = catchswitch within none [label %catch] unwind to caller

catch:
  %catch.pad = catchpad within %catch.switch [i8* null, i32 0, i8* null]
  catchret from %catch.pad to label %try.cont

try.cont:
  ret i32 0
}

; CHECK-LABEL: define void @f10(
define void @f10(i32 %V) personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @g()
          to label %unreachable unwind label %cleanup
; CHECK:       call void @g()
; CHECK-NEXT:  unreachable

unreachable:
  unreachable

cleanup:
  %cp = cleanuppad within none []
  switch i32 %V, label %cleanupret1 [
    i32 0, label %cleanupret2
  ]

cleanupret1:
  cleanupret from %cp unwind to caller

cleanupret2:
  cleanupret from %cp unwind to caller
}

%struct.S = type { i8 }
%struct.S2 = type { i8 }
declare void @"\01??1S2@@QEAA@XZ"(%struct.S2*)
declare void @g()
declare void @use_x(i32 %x)

declare i32 @__CxxFrameHandler3(...)

declare void @llvm.lifetime.start(i64, i8* nocapture)
declare void @llvm.lifetime.end(i64, i8* nocapture)
