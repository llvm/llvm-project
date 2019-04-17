; RUN: opt < %s -inline -argpromotion -instcombine -disable-output

; This test was failing because the inliner would inline @list_DeleteElement
; into @list_DeleteDuplicates and then into @inf_GetBackwardPartnerLits,
; turning the indirect call into a direct one.  This allowed instcombine to see
; the bitcast and eliminate it, deleting the original call and introducing
; another one.  This crashed the inliner because the new call was not in the
; callgraph.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin10.0"


define void @list_DeleteElement(i32 (i8*, i8*)* nocapture %Test) nounwind ssp {
entry:
  %0 = call i32 %Test(i8* null, i8* undef) nounwind
  ret void
}


define void @list_DeleteDuplicates(i32 (i8*, i8*)* nocapture %Test) nounwind ssp {
foo:
  call void @list_DeleteElement(i32 (i8*, i8*)* %Test) nounwind ssp 
  call fastcc void @list_Rplacd1284() nounwind ssp
  unreachable

}

define internal i32 @inf_LiteralsHaveSameSubtermAndAreFromSameClause(i32* nocapture %L1, i32* nocapture %L2) nounwind readonly ssp {
entry:
  unreachable
}


define internal fastcc void @inf_GetBackwardPartnerLits(i32* nocapture %Flags) nounwind ssp {
test:
  call void @list_DeleteDuplicates(i32 (i8*, i8*)* bitcast (i32 (i32*, i32*)* @inf_LiteralsHaveSameSubtermAndAreFromSameClause to i32 (i8*, i8*)*)) nounwind 
  ret void
}


define void @inf_BackwardEmptySortPlusPlus() nounwind ssp {
entry:
  call fastcc void @inf_GetBackwardPartnerLits(i32* null) nounwind ssp
  unreachable
}

define void @inf_BackwardWeakening() nounwind ssp {
entry:
  call fastcc void @inf_GetBackwardPartnerLits(i32* null) nounwind ssp
  unreachable
}

declare fastcc void @list_Rplacd1284() nounwind ssp




;============================
; PR5208

define void @AAA() personality i32 (...)* @__gxx_personality_v0 {
entry:
  %A = alloca i8, i32 undef, align 1
  invoke fastcc void @XXX()
          to label %invcont98 unwind label %lpad156 

invcont98:                          
  unreachable

lpad156:                            
  %exn = landingpad {i8*, i32}
            cleanup
  unreachable
}

declare i32 @__gxx_personality_v0(...)

declare fastcc void @YYY()

define internal fastcc void @XXX() personality i32 (...)* @__gxx_personality_v0 {
entry:
  %B = alloca i8, i32 undef, align 1
  invoke fastcc void @YYY()
          to label %bb260 unwind label %lpad

bb260:                              
  ret void

lpad:                               
  %exn = landingpad {i8*, i32}
            cleanup
  resume { i8*, i32 } %exn
}



;; This exposed a crash handling devirtualized calls.
define void @f1(void ()* %f) ssp {
entry:
  call void %f()
  ret void
}

define void @f4(i32 %size) ssp personality i32 (...)* @__gxx_personality_v0 {
entry:
  invoke void @f1(void ()* @f3)
          to label %invcont3 unwind label %lpad18

invcont3:                                         ; preds = %bb1
  ret void

lpad18:                                           ; preds = %invcont3, %bb1
  %exn = landingpad {i8*, i32}
            cleanup
  unreachable
}

define void @f3() ssp {
entry:
  unreachable
}

declare void @f5() ssp



