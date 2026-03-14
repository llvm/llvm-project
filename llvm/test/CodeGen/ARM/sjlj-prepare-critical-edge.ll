; RUN: llc < %s -O1 -mtriple thumbv7-apple-ios6 | FileCheck %s
; Just make sure no one tries to make the assumption that the normal edge of an
; invoke is never a critical edge.  Previously, this code would assert.

%struct.__CFString = type opaque

declare void @bar(ptr, ptr)

define noalias ptr @foo(ptr nocapture %inRefURL) noreturn ssp personality ptr @__gxx_personality_sj0 {
entry:
  %call = tail call ptr @bar3()
  %call2 = invoke ptr @bar2()
          to label %for.cond unwind label %lpad

for.cond:                                         ; preds = %entry, %for.cond
  invoke void @bar(ptr undef, ptr null)
          to label %for.cond unwind label %lpad5

lpad:                                             ; preds = %entry
  %0 = landingpad { ptr, i32 }
          cleanup
  %1 = extractvalue { ptr, i32 } %0, 0
  %2 = extractvalue { ptr, i32 } %0, 1
  br label %ehcleanup

lpad5:                                            ; preds = %for.cond
  %3 = landingpad { ptr, i32 }
          cleanup
  %4 = extractvalue { ptr, i32 } %3, 0
  %5 = extractvalue { ptr, i32 } %3, 1
  invoke void @release(ptr %call2)
          to label %ehcleanup unwind label %terminate.lpad.i.i16

terminate.lpad.i.i16:                             ; preds = %lpad5
  %6 = landingpad { ptr, i32 }
          catch ptr null
  tail call void @terminatev() noreturn nounwind
  unreachable

ehcleanup:                                        ; preds = %lpad5, %lpad
  %exn.slot.0 = phi ptr [ %1, %lpad ], [ %4, %lpad5 ]
  %ehselector.slot.0 = phi i32 [ %2, %lpad ], [ %5, %lpad5 ]
  invoke void @release(ptr %call)
          to label %_ZN5SmartIPK10__CFStringED1Ev.exit unwind label %terminate.lpad.i.i

terminate.lpad.i.i:                               ; preds = %ehcleanup
  %7 = landingpad { ptr, i32 }
          catch ptr null
  tail call void @terminatev() noreturn nounwind
  unreachable

_ZN5SmartIPK10__CFStringED1Ev.exit:               ; preds = %ehcleanup
  %lpad.val = insertvalue { ptr, i32 } undef, ptr %exn.slot.0, 0
  %lpad.val12 = insertvalue { ptr, i32 } %lpad.val, i32 %ehselector.slot.0, 1
  resume { ptr, i32 } %lpad.val12
}

declare ptr @bar3()

declare ptr @bar2()

declare i32 @__gxx_personality_sj0(...)

declare void @release(ptr)

declare void @terminatev()

; Make sure that the instruction DemoteRegToStack inserts to reload
; %call.i.i.i14.i.i follows the instruction that saves the value to the stack in
; basic block %entry.do.body.i.i.i_crit_edge.
; Previously, DemoteRegToStack would insert a load instruction into the entry
; block to reload %call.i.i.i14.i.i before the phi instruction (%0) in block
; %do.body.i.i.i.

; CHECK-LABEL: __Z4foo1c:
; CHECK: bl __Znwm
; CHECK: {{.*}}@ %do.body.i.i.i.preheader
; CHECK: str r0, [sp, [[OFFSET:#[0-9]+]]]
; CHECK: {{.*}}@ %do.body.i.i.i
; CHECK: ldr [[R0:r[0-9]+]], [sp, [[OFFSET]]]
; CHECK: cbz [[R0]]

%"class.std::__1::basic_string" = type { %"class.std::__1::__compressed_pair" }
%"class.std::__1::__compressed_pair" = type { %"class.std::__1::__libcpp_compressed_pair_imp" }
%"class.std::__1::__libcpp_compressed_pair_imp" = type { %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__rep" }
%"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__rep" = type { %union.anon }
%union.anon = type { %"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__long" }
%"struct.std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::__long" = type { i32, i32, ptr }

@.str = private unnamed_addr constant [12 x i8] c"some_string\00", align 1

define void @_Z4foo1c(i8 signext %a) personality ptr @__gxx_personality_sj0 {
entry:
  %s1 = alloca %"class.std::__1::basic_string", align 4
  call void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6__initEPKcm(ptr %s1, ptr @.str, i32 11)
  %call.i.i.i14.i.i = invoke noalias ptr @_Znwm(i32 1024)
          to label %do.body.i.i.i unwind label %lpad.body

do.body.i.i.i:                                    ; preds = %entry, %_ZNSt3__116allocator_traitsINS_9allocatorIcEEE9constructIccEEvRS2_PT_RKT0_.exit.i.i.i
  %lsr.iv = phi i32 [ %lsr.iv.next, %_ZNSt3__116allocator_traitsINS_9allocatorIcEEE9constructIccEEvRS2_PT_RKT0_.exit.i.i.i ], [ -1024, %entry ]
  %0 = phi ptr [ %incdec.ptr.i.i.i, %_ZNSt3__116allocator_traitsINS_9allocatorIcEEE9constructIccEEvRS2_PT_RKT0_.exit.i.i.i ], [ %call.i.i.i14.i.i, %entry ]
  %new.isnull.i.i.i.i = icmp eq ptr %0, null
  br i1 %new.isnull.i.i.i.i, label %_ZNSt3__116allocator_traitsINS_9allocatorIcEEE9constructIccEEvRS2_PT_RKT0_.exit.i.i.i, label %new.notnull.i.i.i.i

new.notnull.i.i.i.i:                              ; preds = %do.body.i.i.i
  store i8 %a, ptr %0, align 1
  br label %_ZNSt3__116allocator_traitsINS_9allocatorIcEEE9constructIccEEvRS2_PT_RKT0_.exit.i.i.i

_ZNSt3__116allocator_traitsINS_9allocatorIcEEE9constructIccEEvRS2_PT_RKT0_.exit.i.i.i: ; preds = %new.notnull.i.i.i.i, %do.body.i.i.i
  %1 = phi ptr [ null, %do.body.i.i.i ], [ %0, %new.notnull.i.i.i.i ]
  %incdec.ptr.i.i.i = getelementptr inbounds i8, ptr %1, i32 1
  %lsr.iv.next = add i32 %lsr.iv, 1
  %cmp.i16.i.i = icmp eq i32 %lsr.iv.next, 0
  br i1 %cmp.i16.i.i, label %invoke.cont, label %do.body.i.i.i

invoke.cont:                                      ; preds = %_ZNSt3__116allocator_traitsINS_9allocatorIcEEE9constructIccEEvRS2_PT_RKT0_.exit.i.i.i
  invoke void @_Z4foo2Pci(ptr %call.i.i.i14.i.i, i32 1024)
          to label %invoke.cont5 unwind label %lpad2

invoke.cont5:                                     ; preds = %invoke.cont
  %cmp.i.i.i15 = icmp eq ptr %call.i.i.i14.i.i, null
  br i1 %cmp.i.i.i15, label %invoke.cont6, label %_ZNSt3__113__vector_baseIcNS_9allocatorIcEEE5clearEv.exit.i.i.i19

_ZNSt3__113__vector_baseIcNS_9allocatorIcEEE5clearEv.exit.i.i.i19: ; preds = %invoke.cont5
  call void @_ZdlPv(ptr %call.i.i.i14.i.i)
  br label %invoke.cont6

invoke.cont6:                                     ; preds = %_ZNSt3__113__vector_baseIcNS_9allocatorIcEEE5clearEv.exit.i.i.i19, %invoke.cont5
  %call10 = call ptr @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEED1Ev(ptr %s1)
  ret void

lpad.body:                                        ; preds = %entry
  %2 = landingpad { ptr, i32 }
          cleanup
  %3 = extractvalue { ptr, i32 } %2, 0
  %4 = extractvalue { ptr, i32 } %2, 1
  br label %ehcleanup

lpad2:                                            ; preds = %invoke.cont
  %5 = landingpad { ptr, i32 }
          cleanup
  %6 = extractvalue { ptr, i32 } %5, 0
  %7 = extractvalue { ptr, i32 } %5, 1
  %cmp.i.i.i21 = icmp eq ptr %call.i.i.i14.i.i, null
  br i1 %cmp.i.i.i21, label %ehcleanup, label %_ZNSt3__113__vector_baseIcNS_9allocatorIcEEE5clearEv.exit.i.i.i26

_ZNSt3__113__vector_baseIcNS_9allocatorIcEEE5clearEv.exit.i.i.i26: ; preds = %lpad2
  call void @_ZdlPv(ptr %call.i.i.i14.i.i)
  br label %ehcleanup

ehcleanup:                                        ; preds = %_ZNSt3__113__vector_baseIcNS_9allocatorIcEEE5clearEv.exit.i.i.i26, %lpad2, %lpad.body
  %exn.slot.0 = phi ptr [ %3, %lpad.body ], [ %6, %lpad2 ], [ %6, %_ZNSt3__113__vector_baseIcNS_9allocatorIcEEE5clearEv.exit.i.i.i26 ]
  %ehselector.slot.0 = phi i32 [ %4, %lpad.body ], [ %7, %lpad2 ], [ %7, %_ZNSt3__113__vector_baseIcNS_9allocatorIcEEE5clearEv.exit.i.i.i26 ]
  %call12 = invoke ptr @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEED1Ev(ptr %s1)
          to label %eh.resume unwind label %terminate.lpad

eh.resume:                                        ; preds = %ehcleanup
  %lpad.val = insertvalue { ptr, i32 } undef, ptr %exn.slot.0, 0
  %lpad.val13 = insertvalue { ptr, i32 } %lpad.val, i32 %ehselector.slot.0, 1
  resume { ptr, i32 } %lpad.val13

terminate.lpad:                                   ; preds = %ehcleanup
  %8 = landingpad { ptr, i32 }
          catch ptr null
  %9 = extractvalue { ptr, i32 } %8, 0
  call void @__clang_call_terminate(ptr %9)
  unreachable
}

declare void @_Z4foo2Pci(ptr, i32)

define linkonce_odr hidden void @__clang_call_terminate(ptr) {
  %2 = tail call ptr @__cxa_begin_catch(ptr %0)
  tail call void @_ZSt9terminatev()
  unreachable
}

declare ptr @__cxa_begin_catch(ptr)
declare void @_ZSt9terminatev()
declare ptr @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEED1Ev(ptr returned)
declare void @_ZdlPv(ptr) #3
declare noalias ptr @_Znwm(i32)
declare void @_ZNSt3__112basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEE6__initEPKcm(ptr, ptr, i32)
declare void @_Unwind_SjLj_Register(ptr)
declare void @_Unwind_SjLj_Unregister(ptr)
declare ptr @llvm.frameaddress(i32)
declare ptr @llvm.stacksave()
declare void @llvm.stackrestore(ptr)
declare i32 @llvm.eh.sjlj.setjmp(ptr)
declare ptr @llvm.eh.sjlj.lsda()
declare void @llvm.eh.sjlj.callsite(i32)
declare void @llvm.eh.sjlj.functioncontext(ptr)
