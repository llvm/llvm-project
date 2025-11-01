; RUN: llc %s -stop-after=codegenprepare -o - | FileCheck %s
; REQUIRES: x86-registered-target

; CHECK: indirectbr ptr %jumpAddrLoaded, [label %returnExit]

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone sspreq willreturn
define i64 @_sub_split_main__split0(ptr %Result, ptr %_context, ptr noalias %rc, ptr noalias %rc2, ptr %jumpAddr) #0 {
_sub_split_main__split0_entry:
  %RC = alloca i64, align 8
  %RCLineNumber_l5 = alloca i32, align 4
  %Result_l7_c9 = alloca i32, align 4
  br label %_sub_split_main__split0_first

_sub_split_main__split0_first:                    ; preds = %_sub_split_main__split0_entry
  store i64 0, ptr %RC, align 8
  store i32 0, ptr %RCLineNumber_l5, align 4
  %value_l6_c15 = load ptr, ptr %rc, align 8
  %rc3 = call i64 @"RefCount~_copyctor~RefCount~Void"(ptr %rc2, ptr %value_l6_c15, ptr %_context)
  %rcCmp_l6 = icmp ne i64 %rc3, 0
  br i1 %rcCmp_l6, label %l3_chain_unwind, label %l6_rcc_ok

l3_chain_unwind:                                  ; preds = %l6_rcc_ok, %_sub_split_main__split0_first
  %RC1 = phi i64 [ %rc3, %_sub_split_main__split0_first ], [ %rc_l7, %l6_rcc_ok ]
  %RCLineNumber = phi i32 [ 6, %_sub_split_main__split0_first ], [ 7, %l6_rcc_ok ]
  store i64 %RC1, ptr %RC, align 8
  store i32 %RCLineNumber, ptr %RCLineNumber_l5, align 4
  br label %_sub_split_main__split0_rcc_unwind_top

l6_rcc_ok:                                        ; preds = %_sub_split_main__split0_first
  %rc_l7 = call i64 @"RefCount~rc~Int32"(ptr %Result_l7_c9, ptr %rc2)
  %_l7_c9 = load i32, ptr %Result_l7_c9, align 4
  store i32 %_l7_c9, ptr %Result, align 4
  store ptr blockaddress(@main, %returnExit), ptr %jumpAddr, align 8
  %rcCmp_l7_i1 = icmp ne i64 %rc_l7, 0
  br i1 %rcCmp_l7_i1, label %l3_chain_unwind, label %l7_rcc_ok

l7_rcc_ok:                                        ; preds = %l6_rcc_ok
  br label %_sub_split_main__split0_return

_sub_split_main__split0_return:                   ; preds = %l7_rcc_ok, %_sub_split_main__split0_rcc_unwind_top
  %RCValue = load i64, ptr %RC, align 8
  ret i64 %RCValue

_sub_split_main__split0_rcc_unwind_top:           ; preds = %l3_chain_unwind
  call void @stackTracePush()
  br label %_sub_split_main__split0_return
}

; Function Attrs: noinline nounwind optnone sspreq willreturn
define i64 @main(ptr %Result, ptr %_context) #0 {
main_entry:
  %RC = alloca i64, align 8
  %RCLineNumber_l3 = alloca i32, align 4
  %rc_l4 = alloca ptr, align 8
  %rc2_l6 = alloca ptr, align 8
  %jmpAddr_l5_c2 = alloca ptr, align 8
  br label %main_first

main_first:                                       ; preds = %main_entry
  store i64 0, ptr %RC, align 8
  store i32 0, ptr %RCLineNumber_l3, align 4
  store ptr null, ptr %rc_l4, align 8
  store ptr null, ptr %rc2_l6, align 8
  %rc_l43 = call i64 @"RefCount~_ctor~Void"(ptr %rc_l4)
  %rcCmp_l4_i1 = icmp ne i64 %rc_l43, 0
  br i1 %rcCmp_l4_i1, label %l3_chain_unwind, label %l4_rcc_ok

l0_chain_unwind:                                  ; preds = %l3_chain_unwind
  %RC1 = phi i64 [ %RC4, %l3_chain_unwind ]
  %RCLineNumber = phi i32 [ %RCLineNumber5, %l3_chain_unwind ]
  store i64 %RC1, ptr %RC, align 8
  store i32 %RCLineNumber, ptr %RCLineNumber_l3, align 4
  br label %main_rcc_unwind_top

l3_chain_unwind:                                  ; preds = %l4_rcc_ok, %main_first
  %RC4 = phi i64 [ %rc_l43, %main_first ], [ %rc_l5_i3, %l4_rcc_ok ]
  %RCLineNumber5 = phi i32 [ 4, %main_first ], [ 5, %l4_rcc_ok ]
  store i64 %RC4, ptr %RC, align 8
  store i32 %RCLineNumber5, ptr %RCLineNumber_l3, align 4
  %rc2_l6_unw_e = load ptr, ptr %rc2_l6, align 8
  call void @"RefCount~_dtor~Void"(ptr %rc2_l6_unw_e, ptr %_context)
  store ptr null, ptr %rc2_l6, align 8
  %rc_l4_unw_e = load ptr, ptr %rc_l4, align 8
  call void @"RefCount~_dtor~Void"(ptr %rc_l4_unw_e, ptr %_context)
  store ptr null, ptr %rc_l4, align 8
  br label %l0_chain_unwind

l4_rcc_ok:                                        ; preds = %main_first
  store ptr blockaddress(@main, %l5_regularExit), ptr %jmpAddr_l5_c2, align 8
  %rc_l5_i3 = call i64 @_sub_split_main__split0(ptr %Result, ptr %_context, ptr %rc_l4, ptr %rc2_l6, ptr %jmpAddr_l5_c2)
  %rcCmp_l5_i4 = icmp ne i64 %rc_l5_i3, 0
  br i1 %rcCmp_l5_i4, label %l3_chain_unwind, label %l5_rcc_ok

returnExit:                                       ; preds = %l5_rcc_ok
  %rc2_l6_unw_r = load ptr, ptr %rc2_l6, align 8
  call void @"RefCount~_dtor~Void"(ptr %rc2_l6_unw_r, ptr %_context)
  store ptr null, ptr %rc2_l6, align 8
  %rc_l4_unw_r = load ptr, ptr %rc_l4, align 8
  call void @"RefCount~_dtor~Void"(ptr %rc_l4_unw_r, ptr %_context)
  store ptr null, ptr %rc_l4, align 8
  br label %main_return

l5_regularExit:                                   ; No predecessors!
  %rc2_l6_unw = load ptr, ptr %rc2_l6, align 8
  call void @"RefCount~_dtor~Void"(ptr %rc2_l6_unw, ptr %_context)
  store ptr null, ptr %rc2_l6, align 8
  %rc_l4_unw = load ptr, ptr %rc_l4, align 8
  call void @"RefCount~_dtor~Void"(ptr %rc_l4_unw, ptr %_context)
  store ptr null, ptr %rc_l4, align 8
  br label %main_return

l5_rcc_ok:                                        ; preds = %l4_rcc_ok
  %jumpAddrLoaded = load ptr, ptr %jmpAddr_l5_c2, align 8
  indirectbr ptr %jumpAddrLoaded, [label %returnExit]

main_return:                                      ; preds = %l5_regularExit, %returnExit, %main_rcc_unwind_top
  %RCValue = load i64, ptr %RC, align 8
  ret i64 %RCValue

main_rcc_unwind_top:                              ; preds = %l0_chain_unwind
  call void @stackTracePush()
  br label %main_return
}

; Function Attrs: nounwind sspreq willreturn
declare void @stackTracePush() #1

; Function Attrs: nounwind sspreq willreturn
declare i64 @"RefCount~_ctor~Void"(ptr) #1

; Function Attrs: nounwind sspreq willreturn
declare i64 @"RefCount~_copyctor~RefCount~Void"(ptr, ptr, ptr) #1

; Function Attrs: nounwind sspreq willreturn
declare i64 @"RefCount~rc~Int32"(ptr, ptr) #1

; Function Attrs: nounwind sspreq willreturn
declare void @"RefCount~_dtor~Void"(ptr, ptr) #1

attributes #0 = { noinline nounwind optnone sspreq willreturn "frame-pointer"="all" }
attributes #1 = { nounwind sspreq willreturn "frame-pointer"="all" }
