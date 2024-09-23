; RUN: opt -S -passes=cgscc(devirt<4>(inline,function<eager-inv;no-rerun>(early-cse<memssa>,simplifycfg<bonus-inst-threshold=1;no-forward-switch-cond;switch-range-to-icmp;no-switch-to-lookup;keep-loops;no-hoist-common-insts;no-hoist-loads-stores-with-cond-faulting;no-sink-common-insts;speculate-blocks;simplify-cond-branch;no-speculate-unpredictables>,loop-mssa(licm<allowspeculation>)))) < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"struct.__gnu_cxx::__ops::_Iter_equals_val" = type { ptr }

declare ptr @_ZN4llvm16itanium_demangle12OutputBuffer9getBufferEv()

define i64 @_ZNKSt17basic_string_viewIcSt11char_traitsIcEE4sizeEv() {
entry:
  ret i64 0
}

define fastcc i1 @_ZL14decodePunycodeSt17basic_string_viewIcSt11char_traitsIcEERN4llvm16itanium_demangle12OutputBufferE(i64 %InputIdx.2, i1 %cmp23.not) {
entry:
  %call = call i64 @_ZNK4llvm16itanium_demangle12OutputBuffer18getCurrentPositionEv()
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  br i1 true, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.cond
  br label %if.end19

for.body:                                         ; preds = %for.cond
  br label %for.cond

for.cond6:                                        ; preds = %for.inc15
  br i1 false, label %for.end17, label %for.body8

for.body8:                                        ; preds = %for.cond6
  %call10 = call i1 @_ZL7isValidc()
  br label %if.end12

if.end12:                                         ; preds = %for.body8
  br label %arrayinit.body

arrayinit.body:                                   ; preds = %arrayinit.body, %if.end12
  br label %arrayinit.body

arrayinit.end13:                                  ; No predecessors!
  br label %for.inc15

for.inc15:                                        ; preds = %arrayinit.end13
  br label %for.cond6

for.end17:                                        ; preds = %for.cond6
  br label %if.end19

if.end19:                                         ; preds = %for.end17, %for.cond.cleanup
  br label %for.cond21

for.cond21:                                       ; preds = %for.inc100, %if.end19
  br i1 %cmp23.not, label %cleanup102, label %for.body25

for.body25:                                       ; preds = %for.cond21
  br label %for.cond27

for.cond27:                                       ; preds = %for.inc67, %for.body25
  %call30 = call i64 @_ZNKSt17basic_string_viewIcSt11char_traitsIcEE4sizeEv()
  br i1 %cmp23.not, label %cleanup69, label %if.end33

if.end33:                                         ; preds = %for.cond27
  br label %if.end39

if.end39:                                         ; preds = %if.end33
  br i1 false, label %cleanup63, label %if.end42

if.end42:                                         ; preds = %if.end39
  br label %if.else

if.then44:                                        ; No predecessors!
  br label %if.end51

if.else:                                          ; preds = %if.end42
  br label %if.end51

if.end51:                                         ; preds = %if.else, %if.then44
  br label %cleanup63

if.end54:                                         ; No predecessors!
  br label %cleanup63

cleanup63:                                        ; preds = %if.end54, %if.end51, %if.end39
  br label %for.inc67

for.inc67:                                        ; preds = %cleanup63
  br label %for.cond27

cleanup69:                                        ; preds = %for.cond27
  br label %for.end71

for.end71:                                        ; preds = %cleanup69
  %call72 = call i64 @_ZNK4llvm16itanium_demangle12OutputBuffer18getCurrentPositionEv()
  %call77 = call fastcc i64 @"_ZZL14decodePunycodeSt17basic_string_viewIcSt11char_traitsIcEERN4llvm16itanium_demangle12OutputBufferEENK3$_0clEmm"()
  br i1 false, label %cleanup95, label %if.end82

if.end82:                                         ; preds = %for.end71
  %call87 = call fastcc i1 @_ZL10encodeUTF8mPc(i64 %InputIdx.2, ptr null)
  br i1 %call87, label %if.end89, label %cleanup93

if.end89:                                         ; preds = %if.end82
  call void null(ptr null, i64 0, ptr null, i64 0)
  br label %cleanup93

cleanup93:                                        ; preds = %if.end89, %if.end82
  br label %cleanup95

cleanup95:                                        ; preds = %cleanup93, %for.end71
  br label %for.inc100

for.inc100:                                       ; preds = %cleanup95
  br label %for.cond21

cleanup102:                                       ; preds = %for.cond21
  br label %for.end104

for.end104:                                       ; preds = %cleanup102
  call fastcc void @_ZL15removeNullBytesRN4llvm16itanium_demangle12OutputBufferEm()
  br label %cleanup105

cleanup105:                                       ; preds = %for.end104
  br label %cleanup113

cleanup113:                                       ; preds = %cleanup105
  ret i1 false
}

define i64 @_ZNK4llvm16itanium_demangle12OutputBuffer18getCurrentPositionEv() {
entry:
  ret i64 0
}

define i1 @_ZL7isValidc() {
entry:
  unreachable
}

define fastcc i64 @"_ZZL14decodePunycodeSt17basic_string_viewIcSt11char_traitsIcEERN4llvm16itanium_demangle12OutputBufferEENK3$_0clEmm"() {
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %entry
  %Delta.addr.0 = phi i64 [ 0, %entry ], [ 1, %while.cond ]
  %cmp = icmp ugt i64 %Delta.addr.0, 0
  br i1 %cmp, label %while.cond, label %while.end

while.end:                                        ; preds = %while.cond
  ret i64 0
}

define fastcc i1 @_ZL10encodeUTF8mPc(i64 %CodePoint, ptr %Output) {
entry:
  %0 = and i64 %CodePoint, 1
  %or.cond = icmp eq i64 %0, 0
  br i1 %or.cond, label %return, label %if.end

if.end:                                           ; preds = %entry
  %cmp2 = icmp ult i64 %CodePoint, 1
  br i1 %cmp2, label %if.then3, label %if.end4

if.then3:                                         ; preds = %if.end
  store i8 0, ptr %Output, align 1
  br label %return

if.end4:                                          ; preds = %if.end
  %cmp5 = icmp ult i64 %CodePoint, 2048
  br i1 %cmp5, label %return, label %if.end13

if.end13:                                         ; preds = %if.end4
  %cmp14 = icmp ult i64 %CodePoint, 65536
  br i1 %cmp14, label %if.then15, label %if.end29

if.then15:                                        ; preds = %if.end13
  %shr16 = lshr i64 %CodePoint, 1
  %1 = trunc i64 %shr16 to i8
  br label %return

if.end29:                                         ; preds = %if.end13
  %cmp30 = icmp ult i64 %CodePoint, 1114112
  %spec.select = select i1 %cmp30, i1 false, i1 false
  br label %return

return:                                           ; preds = %if.end29, %if.then15, %if.end4, %if.then3, %entry
  %retval.0 = phi i1 [ false, %if.then3 ], [ true, %if.then15 ], [ false, %entry ], [ false, %if.end29 ], [ true, %if.end4 ]
  ret i1 %retval.0
}

define fastcc void @_ZL15removeNullBytesRN4llvm16itanium_demangle12OutputBufferEm() {
entry:
  %ref.tmp = alloca i8, align 1
  %call = call ptr @_ZN4llvm16itanium_demangle12OutputBuffer9getBufferEv()
  store i8 0, ptr %ref.tmp, align 1
  %call3 = call ptr @_ZSt6removeIPccET_S1_S1_RKT0_(ptr %call)
  ret void
}

define ptr @_ZSt6removeIPccET_S1_S1_RKT0_(ptr %__last) {
entry:
  %call2 = call ptr @_ZSt11__remove_ifIPcN9__gnu_cxx5__ops16_Iter_equals_valIKcEEET_S6_S6_T0_(ptr %__last)
  ret ptr null
}

define ptr @_ZSt11__remove_ifIPcN9__gnu_cxx5__ops16_Iter_equals_valIKcEEET_S6_S6_T0_(ptr %__last) {
entry:
  %__pred = alloca %"struct.__gnu_cxx::__ops::_Iter_equals_val", align 8
  store ptr null, ptr %__pred, align 8
  %call = call ptr @_ZSt9__find_ifIPcN9__gnu_cxx5__ops16_Iter_equals_valIKcEEET_S6_S6_T0_()
  %cmp = icmp eq ptr %call, null
  br i1 %cmp, label %return, label %for.cond

for.cond:                                         ; preds = %if.then4, %for.body, %entry
  %call.pn = phi ptr [ %__last, %entry ], [ null, %if.then4 ], [ null, %for.body ]
  %cmp2.not = icmp eq ptr %call.pn, null
  br i1 %cmp2.not, label %return, label %for.body

for.body:                                         ; preds = %for.cond
  %call3 = load i1, ptr null, align 1
  br i1 %call3, label %for.cond, label %if.then4

if.then4:                                         ; preds = %for.body
  store i8 0, ptr %__last, align 1
  br label %for.cond

return:                                           ; preds = %for.cond, %entry
  ret ptr null
}

define ptr @_ZSt9__find_ifIPcN9__gnu_cxx5__ops16_Iter_equals_valIKcEEET_S6_S6_T0_() {
entry:
  %call1 = call ptr @_ZSt9__find_ifIPcN9__gnu_cxx5__ops16_Iter_equals_valIKcEEET_S6_S6_T0_St26random_access_iterator_tag(ptr null)
  ret ptr %call1
}

define ptr @_ZSt9__find_ifIPcN9__gnu_cxx5__ops16_Iter_equals_valIKcEEET_S6_S6_T0_St26random_access_iterator_tag(ptr %__first.addr.0) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %if.end11, %entry
  %__first.addr.01 = phi ptr [ null, %entry ], [ %incdec.ptr12, %if.end11 ]
  %__trip_count.0 = phi i64 [ 0, %entry ], [ 1, %if.end11 ]
  %cmp = icmp sgt i64 %__trip_count.0, 0
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %call = load i1, ptr %__first.addr.01, align 1
  br i1 %call, label %cleanup, label %if.end

if.end:                                           ; preds = %for.body
  %incdec.ptr = getelementptr i8, ptr %__first.addr.01, i64 1
  %call1 = load i1, ptr %incdec.ptr, align 1
  br i1 %call1, label %cleanup, label %if.end3

if.end3:                                          ; preds = %if.end
  %incdec.ptr4 = getelementptr i8, ptr %__first.addr.0, i64 2
  %call5 = load i1, ptr %incdec.ptr4, align 1
  br i1 %call5, label %cleanup, label %if.end7

if.end7:                                          ; preds = %if.end3
  %call9 = load i1, ptr null, align 1
  br i1 %call9, label %cleanup, label %if.end11

if.end11:                                         ; preds = %if.end7
  %incdec.ptr12 = getelementptr i8, ptr %__first.addr.01, i64 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %sub.ptr.rhs.cast14 = ptrtoint ptr %__first.addr.01 to i64
  switch i64 %sub.ptr.rhs.cast14, label %sw.default [
    i64 -3, label %sw.bb
    i64 1, label %sw.bb20
    i64 0, label %sw.bb25
  ]

sw.bb:                                            ; preds = %for.end
  %call16 = load i1, ptr null, align 1
  br i1 %call16, label %cleanup, label %if.end18

if.end18:                                         ; preds = %sw.bb
  %incdec.ptr19 = getelementptr i8, ptr %__first.addr.01, i64 1
  br label %sw.bb20

sw.bb20:                                          ; preds = %if.end18, %for.end
  %__first.addr.1 = phi ptr [ null, %for.end ], [ %incdec.ptr19, %if.end18 ]
  %incdec.ptr24 = getelementptr i8, ptr %__first.addr.1, i64 1
  br label %sw.bb25

sw.bb25:                                          ; preds = %sw.bb20, %for.end
  %__first.addr.2 = phi ptr [ null, %for.end ], [ %incdec.ptr24, %sw.bb20 ]
  %call26 = load i1, ptr null, align 1
  br i1 %call26, label %cleanup, label %sw.default

sw.default:                                       ; preds = %sw.bb25, %for.end
  br label %cleanup

cleanup:                                          ; preds = %sw.default, %sw.bb25, %sw.bb, %if.end7, %if.end3, %if.end, %for.body
  %retval.0 = phi ptr [ null, %sw.default ], [ null, %for.body ], [ %incdec.ptr, %if.end ], [ null, %if.end3 ], [ null, %if.end7 ], [ null, %sw.bb ], [ %__first.addr.2, %sw.bb25 ]
  ret ptr %retval.0
}

; uselistorder directives
uselistorder ptr @_ZNK4llvm16itanium_demangle12OutputBuffer18getCurrentPositionEv, { 1, 0 }

