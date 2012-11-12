; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=core2 | FileCheck %s
; Reduced from JavaScriptCore

%"class.JSC::CodeLocationCall" = type { [8 x i8] }
%"class.JSC::JSGlobalData" = type { [4 x i8] }
%"class.JSC::FunctionPtr" = type { i8* }
%"class.JSC::Structure" = type { [4 x i8] }
%"class.JSC::UString" = type { i8* }
%"class.JSC::JSString" = type { [16 x i8], i32, %"class.JSC::UString", i32 }

declare hidden fastcc void @_ZN3JSCL23returnToThrowTrampolineEPNS_12JSGlobalDataENS_16ReturnAddressPtrERS2_(%"class.JSC::JSGlobalData"* nocapture, i8*, %"class.JSC::FunctionPtr"* nocapture) nounwind noinline ssp

; Avoid hoisting the test above loads or copies
; CHECK: %entry
; CHECK: test
; CHECK-NOT: mov
; CHECK: je
define i32 @cti_op_eq(i8** nocapture %args) nounwind ssp {
entry:
  %0 = load i8** null, align 8
  %tmp13 = bitcast i8* %0 to %"class.JSC::CodeLocationCall"*
  %tobool.i.i.i = icmp ugt i8* undef, inttoptr (i64 281474976710655 to i8*)
  %or.cond.i = and i1 %tobool.i.i.i, undef
  br i1 %or.cond.i, label %if.then.i, label %if.end.i

if.then.i:                                        ; preds = %entry
  br i1 undef, label %if.then.i.i.i, label %_ZN3JSC7JSValue19equalSlowCaseInlineEPNS_9ExecStateES0_S0_.exit

if.then.i.i.i:                                    ; preds = %if.then.i
  %conv.i.i.i.i = trunc i64 undef to i32
  br label %_ZN3JSC7JSValue19equalSlowCaseInlineEPNS_9ExecStateES0_S0_.exit

if.end.i:                                         ; preds = %entry
  br i1 undef, label %land.rhs.i121.i, label %_ZNK3JSC7JSValue8isStringEv.exit122.i

land.rhs.i121.i:                                  ; preds = %if.end.i
  %tmp.i.i117.i = load %"class.JSC::Structure"** undef, align 8
  br label %_ZNK3JSC7JSValue8isStringEv.exit122.i

_ZNK3JSC7JSValue8isStringEv.exit122.i:            ; preds = %land.rhs.i121.i, %if.end.i
  %brmerge.i = or i1 undef, false
  %or.cond = or i1 false, %brmerge.i
  br i1 %or.cond, label %_ZN3JSC7JSValue19equalSlowCaseInlineEPNS_9ExecStateES0_S0_.exit, label %if.then.i92.i

if.then.i92.i:                                    ; preds = %_ZNK3JSC7JSValue8isStringEv.exit122.i
  tail call void @_ZNK3JSC8JSString11resolveRopeEPNS_9ExecStateE(%"class.JSC::JSString"* undef, %"class.JSC::CodeLocationCall"* %tmp13) nounwind
  unreachable

_ZN3JSC7JSValue19equalSlowCaseInlineEPNS_9ExecStateES0_S0_.exit: ; preds = %_ZNK3JSC7JSValue8isStringEv.exit122.i, %if.then.i.i.i, %if.then.i

  %1 = load i8** undef, align 8
  br i1 undef, label %do.end39, label %do.body27

do.body27:                                        ; preds = %_ZN3JSC7JSValue19equalSlowCaseInlineEPNS_9ExecStateES0_S0_.exit
  %tmp30 = bitcast i8* %1 to %"class.JSC::JSGlobalData"*
  %2 = getelementptr inbounds i8** %args, i64 -1
  %3 = bitcast i8** %2 to %"class.JSC::FunctionPtr"*
  tail call fastcc void @_ZN3JSCL23returnToThrowTrampolineEPNS_12JSGlobalDataENS_16ReturnAddressPtrERS2_(%"class.JSC::JSGlobalData"* %tmp30, i8* undef, %"class.JSC::FunctionPtr"* %3)
  unreachable

do.end39:                                         ; preds = %_ZN3JSC7JSValue19equalSlowCaseInlineEPNS_9ExecStateES0_S0_.exit
  ret i32 undef
}

declare void @_ZNK3JSC8JSString11resolveRopeEPNS_9ExecStateE(%"class.JSC::JSString"*, %"class.JSC::CodeLocationCall"*)
