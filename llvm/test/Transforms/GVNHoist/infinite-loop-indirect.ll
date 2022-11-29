; RUN: opt -S -passes=gvn-hoist < %s | FileCheck %s

; Checking gvn-hoist in case of indirect branches.

; Check that the bitcast is not hoisted because it is after an indirect call
; CHECK-LABEL: @foo
; CHECK-LABEL: l1.preheader:
; CHECK-NEXT: bitcast
; CHECK-LABEL: l1
; CHECK: bitcast

%class.bar = type { i8*, %class.base* }
%class.base = type { i32 (...)** }

@bar = local_unnamed_addr global i32 ()* null, align 8
@bar1 = local_unnamed_addr global i32 ()* null, align 8

define i32 @foo(i32* nocapture readonly %i) {
entry:
  %agg.tmp = alloca %class.bar, align 8
  %x= getelementptr inbounds %class.bar, %class.bar* %agg.tmp, i64 0, i32 1
  %y = load %class.base*, %class.base** %x, align 8
  %0 = load i32, i32* %i, align 4
  %.off = add i32 %0, -1
  %switch = icmp ult i32 %.off, 2
  br i1 %switch, label %l1.preheader, label %sw.default

l1.preheader:                                     ; preds = %sw.default, %entry
  %b1 = bitcast %class.base* %y to void (%class.base*)***
  br label %l1

l1:                                               ; preds = %l1.preheader, %l1
  %1 = load i32 ()*, i32 ()** @bar, align 8
  %call = tail call i32 %1()
  %b2 = bitcast %class.base* %y to void (%class.base*)***
  br label %l1

sw.default:                                       ; preds = %entry
  %2 = load i32 ()*, i32 ()** @bar1, align 8
  %call2 = tail call i32 %2()
  br label %l1.preheader
}


; Any instruction inside an infinite loop will not be hoisted because
; there is no path to exit of the function.

; CHECK-LABEL: @foo1
; CHECK-LABEL: l1.preheader:
; CHECK-NEXT: bitcast
; CHECK-LABEL: l1:
; CHECK: bitcast

define i32 @foo1(i32* nocapture readonly %i) {
entry:
  %agg.tmp = alloca %class.bar, align 8
  %x= getelementptr inbounds %class.bar, %class.bar* %agg.tmp, i64 0, i32 1
  %y = load %class.base*, %class.base** %x, align 8
  %0 = load i32, i32* %i, align 4
  %.off = add i32 %0, -1
  %switch = icmp ult i32 %.off, 2
  br i1 %switch, label %l1.preheader, label %sw.default

l1.preheader:                                     ; preds = %sw.default, %entry
  %b1 = bitcast %class.base* %y to void (%class.base*)***
  %y1 = load %class.base*, %class.base** %x, align 8
  br label %l1

l1:                                               ; preds = %l1.preheader, %l1
  %b2 = bitcast %class.base* %y to void (%class.base*)***
  %1 = load i32 ()*, i32 ()** @bar, align 8
  %y2 = load %class.base*, %class.base** %x, align 8
  %call = tail call i32 %1()
  br label %l1

sw.default:                                       ; preds = %entry
  %2 = load i32 ()*, i32 ()** @bar1, align 8
  %call2 = tail call i32 %2()
  br label %l1.preheader
}

; Check that bitcast is hoisted even when one of them is partially redundant.
; CHECK-LABEL: @test13
; CHECK: bitcast
; CHECK-NOT: bitcast

define i32 @test13(i32* %P, i8* %Ptr, i32* nocapture readonly %i) {
entry:
  %agg.tmp = alloca %class.bar, align 8
  %x= getelementptr inbounds %class.bar, %class.bar* %agg.tmp, i64 0, i32 1
  %y = load %class.base*, %class.base** %x, align 8
  indirectbr i8* %Ptr, [label %BrBlock, label %B2]

B2:
  %b1 = bitcast %class.base* %y to void (%class.base*)***
  store i32 4, i32 *%P
  br label %BrBlock

BrBlock:
  %b2 = bitcast %class.base* %y to void (%class.base*)***
  %L = load i32, i32* %P
  %C = icmp eq i32 %L, 42
  br i1 %C, label %T, label %F

T:
  ret i32 123
F:
  ret i32 1422
}

; Check that the bitcast is not hoisted because anticipability
; cannot be guaranteed here as one of the indirect branch targets
; do not have the bitcast instruction.

; CHECK-LABEL: @test14
; CHECK-LABEL: B2:
; CHECK-NEXT: bitcast
; CHECK-LABEL: BrBlock:
; CHECK-NEXT: bitcast

define i32 @test14(i32* %P, i8* %Ptr, i32* nocapture readonly %i) {
entry:
  %agg.tmp = alloca %class.bar, align 8
  %x= getelementptr inbounds %class.bar, %class.bar* %agg.tmp, i64 0, i32 1
  %y = load %class.base*, %class.base** %x, align 8
  indirectbr i8* %Ptr, [label %BrBlock, label %B2, label %T]

B2:
  %b1 = bitcast %class.base* %y to void (%class.base*)***
  store i32 4, i32 *%P
  br label %BrBlock

BrBlock:
  %b2 = bitcast %class.base* %y to void (%class.base*)***
  %L = load i32, i32* %P
  %C = icmp eq i32 %L, 42
  br i1 %C, label %T, label %F

T:
  %pi = load i32, i32* %i, align 4
  ret i32 %pi
F:
  %pl = load i32, i32* %P
  ret i32 %pl
}


; Check that the bitcast is not hoisted because of a cycle
; due to indirect branches
; CHECK-LABEL: @test16
; CHECK-LABEL: B2:
; CHECK-NEXT: bitcast
; CHECK-LABEL: BrBlock:
; CHECK-NEXT: bitcast

define i32 @test16(i32* %P, i8* %Ptr, i32* nocapture readonly %i) {
entry:
  %agg.tmp = alloca %class.bar, align 8
  %x= getelementptr inbounds %class.bar, %class.bar* %agg.tmp, i64 0, i32 1
  %y = load %class.base*, %class.base** %x, align 8
  indirectbr i8* %Ptr, [label %BrBlock, label %B2]

B2:
  %b1 = bitcast %class.base* %y to void (%class.base*)***
  %0 = load i32, i32* %i, align 4
  store i32 %0, i32 *%P
  br label %BrBlock

BrBlock:
  %b2 = bitcast %class.base* %y to void (%class.base*)***
  %L = load i32, i32* %P
  %C = icmp eq i32 %L, 42
  br i1 %C, label %T, label %F

T:
  indirectbr i32* %P, [label %BrBlock, label %B2]

F:
  indirectbr i8* %Ptr, [label %BrBlock, label %B2]
}


@_ZTIi = external constant i8*

; Check that an instruction is not hoisted out of landing pad (%lpad4)
; Also within a landing pad no redundancies are removed by gvn-hoist,
; however an instruction may be hoisted into a landing pad if
; landing pad has direct branches (e.g., %lpad to %catch1, %catch)
; This CFG has a cycle (%lpad -> %catch1 -> %lpad4 -> %lpad)

; CHECK-LABEL: @foo2
; Check that nothing gets hoisted out of %lpad
; CHECK-LABEL: lpad:
; CHECK: %bc1 = add i32 %0, 10
; CHECK: %bc7 = add i32 %0, 10

; Check that the add is hoisted
; CHECK-LABEL: catch1:
; CHECK-NEXT: invoke

; Check that the add is hoisted
; CHECK-LABEL: catch:
; CHECK-NEXT: load

; Check that other adds are not hoisted
; CHECK-LABEL: lpad4:
; CHECK: %bc5 = add i32 %0, 10
; CHECK-LABEL: unreachable:
; CHECK: %bc2 = add i32 %0, 10

; Function Attrs: noinline uwtable
define i32 @foo2(i32* nocapture readonly %i) local_unnamed_addr personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %0 = load i32, i32* %i, align 4
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %try.cont, label %if.then

if.then:
  %exception = tail call i8* @__cxa_allocate_exception(i64 4) #2
  %1 = bitcast i8* %exception to i32*
  store i32 %0, i32* %1, align 16
  invoke void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null) #3
          to label %unreachable unwind label %lpad

lpad:
  %2 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIi to i8*)
          catch i8* null
  %bc1 = add i32 %0, 10
  %3 = extractvalue { i8*, i32 } %2, 0
  %4 = extractvalue { i8*, i32 } %2, 1
  %5 = tail call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #2
  %matches = icmp eq i32 %4, %5
  %bc7 = add i32 %0, 10
  %6 = tail call i8* @__cxa_begin_catch(i8* %3) #2
  br i1 %matches, label %catch1, label %catch

catch1:
  %bc3 = add i32 %0, 10
  invoke void @__cxa_rethrow() #3
          to label %unreachable unwind label %lpad4

catch:
  %bc4 = add i32 %0, 10
  %7 = load i32, i32* %i, align 4
  %add = add nsw i32 %7, 1
  tail call void @__cxa_end_catch()
  br label %try.cont

lpad4:
  %8 = landingpad { i8*, i32 }
          cleanup
  %bc5 = add i32 %0, 10
  tail call void @__cxa_end_catch() #2
  invoke void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null) #3
          to label %unreachable unwind label %lpad

try.cont:
  %k.0 = phi i32 [ %add, %catch ], [ 0, %entry ]
  %bc6 = add i32 %0, 10
  ret i32 %k.0

unreachable:
  %bc2 = add i32 %0, 10
  ret i32 %bc2
}

declare i8* @__cxa_allocate_exception(i64) local_unnamed_addr

declare void @__cxa_throw(i8*, i8*, i8*) local_unnamed_addr

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nounwind readnone
declare i32 @llvm.eh.typeid.for(i8*) #1

declare i8* @__cxa_begin_catch(i8*) local_unnamed_addr

declare void @__cxa_end_catch() local_unnamed_addr

declare void @__cxa_rethrow() local_unnamed_addr

attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
attributes #3 = { noreturn }
