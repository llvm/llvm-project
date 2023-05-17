; RUN: llc < %s -mtriple=armv4t-unknown-linux-gnueabi  | FileCheck %s
; PR 7433
; XFAIL: *

%0 = type { ptr, ptr }
%1 = type { ptr, ptr, ptr }
%"class.llvm::Record" = type { i32, %"class.std::basic_string", %"class.llvm::SMLoc", %"class.std::vector", %"class.std::vector", %"class.std::vector" }
%"class.llvm::RecordVal" = type { %"class.std::basic_string", ptr, i32, ptr }
%"class.llvm::SMLoc" = type { ptr }
%"class.llvm::StringInit" = type { [8 x i8], %"class.std::basic_string" }
%"class.std::basic_string" = type { %"class.llvm::SMLoc" }
%"class.std::vector" = type { [12 x i8] }
%"struct.llvm::Init" = type { ptr }

@_ZTIN4llvm5RecTyE = external constant %0         ; <ptr> [#uses=1]
@_ZTIN4llvm4InitE = external constant %0          ; <ptr> [#uses=1]
@_ZTIN4llvm11RecordRecTyE = external constant %1  ; <ptr> [#uses=1]
@.str8 = external constant [47 x i8]              ; <ptr> [#uses=1]
@_ZTIN4llvm9UnsetInitE = external constant %1     ; <ptr> [#uses=1]
@.str51 = external constant [45 x i8]             ; <ptr> [#uses=1]
@__PRETTY_FUNCTION__._ZNK4llvm7VarInit12getFieldInitERNS_6RecordEPKNS_9RecordValERKSs = external constant [116 x i8] ; <ptr> [#uses=1]

@_ZN4llvm9RecordValC1ERKSsPNS_5RecTyEj = alias void (ptr, ptr, ptr, i32), ptr @_ZN4llvm9RecordValC2ERKSsPNS_5RecTyEj ; <ptr> [#uses=0]

declare ptr @__dynamic_cast(ptr, ptr, ptr, i32)

declare void @__assert_fail(ptr, ptr, i32, ptr) noreturn

declare void @_ZN4llvm9RecordValC2ERKSsPNS_5RecTyEj(ptr, ptr, ptr, i32) align 2

define ptr @_ZNK4llvm7VarInit12getFieldInitERNS_6RecordEPKNS_9RecordValERKSs(ptr %this, ptr %R, ptr %RV, ptr %FieldName) align 2 {
;CHECK:  ldmia sp!, {r4, r5, r6, r7, r8, lr}
;CHECK:  bx  r12  @ TAILCALL
entry:
  %.loc = alloca i32                              ; <ptr> [#uses=2]
  %tmp.i = getelementptr inbounds %"class.llvm::StringInit", ptr %this, i32 0, i32 0, i32 4 ; <ptr> [#uses=1]
  %tmp2.i = load ptr, ptr %tmp.i        ; <ptr> [#uses=2]
  %0 = icmp eq ptr %tmp2.i, null ; <i1> [#uses=1]
  br i1 %0, label %entry.return_crit_edge, label %tmpbb

entry.return_crit_edge:                           ; preds = %entry
  br label %return

tmpbb:                                            ; preds = %entry
  %1 = tail call ptr @__dynamic_cast(ptr %tmp2.i, ptr @_ZTIN4llvm5RecTyE, ptr @_ZTIN4llvm11RecordRecTyE, i32 -1) ; <ptr> [#uses=1]
  %phitmp = icmp eq ptr %1, null                  ; <i1> [#uses=1]
  br i1 %phitmp, label %.return_crit_edge, label %if.then

.return_crit_edge:                                ; preds = %tmpbb
  br label %return

if.then:                                          ; preds = %tmpbb
  %tmp2.i.i.i.i = getelementptr inbounds %"class.llvm::StringInit", ptr %this, i32 0, i32 1, i32 0, i32 0 ; <ptr> [#uses=1]
  %tmp3.i.i.i.i = load ptr, ptr %tmp2.i.i.i.i         ; <ptr> [#uses=2]
  %arrayidx.i.i.i.i = getelementptr inbounds i8, ptr %tmp3.i.i.i.i, i32 -12 ; <ptr> [#uses=1]
  %tmp2.i.i.i = load i32, ptr %arrayidx.i.i.i.i              ; <i32> [#uses=1]
  %tmp.i5 = getelementptr inbounds %"class.llvm::Record", ptr %R, i32 0, i32 4 ; <ptr> [#uses=1]
  %tmp2.i.i = getelementptr inbounds %"class.llvm::Record", ptr %R, i32 0, i32 4, i32 0, i32 4 ; <ptr> [#uses=1]
  %tmp3.i.i6 = load ptr, ptr %tmp2.i.i ; <ptr> [#uses=1]
  %tmp6.i.i = load ptr, ptr %tmp.i5 ; <ptr> [#uses=5]
  %sub.ptr.lhs.cast.i.i = ptrtoint ptr %tmp3.i.i6 to i32 ; <i32> [#uses=1]
  %sub.ptr.rhs.cast.i.i = ptrtoint ptr %tmp6.i.i to i32 ; <i32> [#uses=1]
  %sub.ptr.sub.i.i = sub i32 %sub.ptr.lhs.cast.i.i, %sub.ptr.rhs.cast.i.i ; <i32> [#uses=1]
  %sub.ptr.div.i.i = ashr i32 %sub.ptr.sub.i.i, 4 ; <i32> [#uses=1]
  br label %codeRepl

codeRepl:                                         ; preds = %if.then
  %targetBlock = call i1 @_ZNK4llvm7VarInit12getFieldInitERNS_6RecordEPKNS_9RecordValERKSs_for.cond.i(i32 %sub.ptr.div.i.i, ptr %tmp6.i.i, i32 %tmp2.i.i.i, ptr %tmp3.i.i.i.i, ptr %.loc) ; <i1> [#uses=1]
  %.reload = load i32, ptr %.loc                      ; <i32> [#uses=3]
  br i1 %targetBlock, label %for.cond.i.return_crit_edge, label %_ZN4llvm6Record8getValueENS_9StringRefE.exit

for.cond.i.return_crit_edge:                      ; preds = %codeRepl
  br label %return

_ZN4llvm6Record8getValueENS_9StringRefE.exit:     ; preds = %codeRepl
  %add.ptr.i.i = getelementptr inbounds %"class.llvm::RecordVal", ptr %tmp6.i.i, i32 %.reload ; <ptr> [#uses=2]
  %tobool5 = icmp eq ptr %add.ptr.i.i, null ; <i1> [#uses=1]
  br i1 %tobool5, label %_ZN4llvm6Record8getValueENS_9StringRefE.exit.return_crit_edge, label %if.then6

_ZN4llvm6Record8getValueENS_9StringRefE.exit.return_crit_edge: ; preds = %_ZN4llvm6Record8getValueENS_9StringRefE.exit
  br label %return

if.then6:                                         ; preds = %_ZN4llvm6Record8getValueENS_9StringRefE.exit
  %cmp = icmp eq ptr %add.ptr.i.i, %RV ; <i1> [#uses=1]
  br i1 %cmp, label %if.then6.if.end_crit_edge, label %land.lhs.true

if.then6.if.end_crit_edge:                        ; preds = %if.then6
  br label %if.end

land.lhs.true:                                    ; preds = %if.then6
  %tobool10 = icmp eq ptr %RV, null ; <i1> [#uses=1]
  br i1 %tobool10, label %lor.lhs.false, label %land.lhs.true.return_crit_edge

land.lhs.true.return_crit_edge:                   ; preds = %land.lhs.true
  br label %return

lor.lhs.false:                                    ; preds = %land.lhs.true
  %tmp.i3 = getelementptr inbounds %"class.llvm::RecordVal", ptr %tmp6.i.i, i32 %.reload, i32 3 ; <ptr> [#uses=1]
  %tmp2.i4 = load ptr, ptr %tmp.i3  ; <ptr> [#uses=2]
  %2 = icmp eq ptr %tmp2.i4, null ; <i1> [#uses=1]
  br i1 %2, label %lor.lhs.false.if.end_crit_edge, label %tmpbb1

lor.lhs.false.if.end_crit_edge:                   ; preds = %lor.lhs.false
  br label %if.end

tmpbb1:                                           ; preds = %lor.lhs.false
  %3 = tail call ptr @__dynamic_cast(ptr %tmp2.i4, ptr @_ZTIN4llvm4InitE, ptr @_ZTIN4llvm9UnsetInitE, i32 -1) ; <ptr> [#uses=1]
  %phitmp32 = icmp eq ptr %3, null                ; <i1> [#uses=1]
  br i1 %phitmp32, label %.if.end_crit_edge, label %.return_crit_edge1

.return_crit_edge1:                               ; preds = %tmpbb1
  br label %return

.if.end_crit_edge:                                ; preds = %tmpbb1
  br label %if.end

if.end:                                           ; preds = %.if.end_crit_edge, %lor.lhs.false.if.end_crit_edge, %if.then6.if.end_crit_edge
  %tmp.i1 = getelementptr inbounds %"class.llvm::RecordVal", ptr %tmp6.i.i, i32 %.reload, i32 3 ; <ptr> [#uses=1]
  %tmp2.i2 = load ptr, ptr %tmp.i1  ; <ptr> [#uses=3]
  %cmp19 = icmp eq ptr %tmp2.i2, %this ; <i1> [#uses=1]
  br i1 %cmp19, label %cond.false, label %cond.end

cond.false:                                       ; preds = %if.end
  tail call void @__assert_fail(ptr @.str51, ptr @.str8, i32 1141, ptr @__PRETTY_FUNCTION__._ZNK4llvm7VarInit12getFieldInitERNS_6RecordEPKNS_9RecordValERKSs) noreturn
  unreachable

cond.end:                                         ; preds = %if.end
  %4 = load ptr, ptr %tmp2.i2 ; <ptr> [#uses=1]
  %vfn = getelementptr inbounds ptr, ptr %4, i32 8 ; <ptr> [#uses=1]
  %5 = load ptr, ptr %vfn ; <ptr> [#uses=1]
  %call25 = tail call ptr %5(ptr %tmp2.i2, ptr %R, ptr %RV, ptr %FieldName) ; <ptr> [#uses=1]
  ret ptr %call25

return:                                           ; preds = %.return_crit_edge1, %land.lhs.true.return_crit_edge, %_ZN4llvm6Record8getValueENS_9StringRefE.exit.return_crit_edge, %for.cond.i.return_crit_edge, %.return_crit_edge, %entry.return_crit_edge
  ret ptr null
}

declare i1 @_ZNK4llvm7VarInit12getFieldInitERNS_6RecordEPKNS_9RecordValERKSs_for.cond.i(i32, ptr, i32, ptr, ptr)
