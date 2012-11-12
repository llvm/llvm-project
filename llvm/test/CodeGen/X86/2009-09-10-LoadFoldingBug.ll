; RUN: llc < %s -mtriple=x86_64-apple-darwin10.0 -relocation-model=pic -disable-fp-elim | FileCheck %s

; It's not legal to fold a load from 32-bit stack slot into a 64-bit
; instruction. If done, the instruction does a 64-bit load and that's not
; safe. This can happen we a subreg_to_reg 0 has been coalesced. One
; exception is when the instruction that folds the load is a move, then we
; can simply turn it into a 32-bit load from the stack slot.
; rdar://7170444

%struct.ComplexType = type { i32 }

define i32 @t(i32 %clientPort, i32 %pluginID, i32 %requestID, i32 %objectID, i64 %serverIdentifier, i64 %argumentsData, i32 %argumentsLength) ssp {
entry:
; CHECK: _t:
; CHECK: movl 16(%rbp),
  %0 = zext i32 %argumentsLength to i64           ; <i64> [#uses=1]
  %1 = zext i32 %clientPort to i64                ; <i64> [#uses=1]
  %2 = inttoptr i64 %1 to %struct.ComplexType*    ; <%struct.ComplexType*> [#uses=1]
  %3 = invoke i8* @pluginInstance(i8* undef, i32 %pluginID)
          to label %invcont unwind label %lpad    ; <i8*> [#uses=1]

invcont:                                          ; preds = %entry
  %4 = add i32 %requestID, %pluginID              ; <i32> [#uses=0]
  %5 = invoke zeroext i8 @invoke(i8* %3, i32 %objectID, i8* undef, i64 %argumentsData, i32 %argumentsLength, i64* undef, i32* undef)
          to label %invcont1 unwind label %lpad   ; <i8> [#uses=0]

invcont1:                                         ; preds = %invcont
  %6 = getelementptr inbounds %struct.ComplexType* %2, i64 0, i32 0 ; <i32*> [#uses=1]
  %7 = load i32* %6, align 4                      ; <i32> [#uses=1]
  invoke void @booleanAndDataReply(i32 %7, i32 undef, i32 %requestID, i32 undef, i64 undef, i32 undef)
          to label %invcont2 unwind label %lpad

invcont2:                                         ; preds = %invcont1
  ret i32 0

lpad:                                             ; preds = %invcont1, %invcont, %entry
  %exn = landingpad {i8*, i32} personality i32 (...)* @__gxx_personality_v0
            cleanup
  %8 = call i32 @vm_deallocate(i32 undef, i64 0, i64 %0) ; <i32> [#uses=0]
  unreachable
}

declare i32 @vm_deallocate(i32, i64, i64)

declare i8* @pluginInstance(i8*, i32)

declare zeroext i8 @invoke(i8*, i32, i8*, i64, i32, i64*, i32*)

declare void @booleanAndDataReply(i32, i32, i32, i32, i64, i32)

declare i32 @__gxx_personality_v0(...)
